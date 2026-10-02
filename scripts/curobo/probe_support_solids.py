"""Probe, on the GPU box, that the planner holds the support surfaces' solids as tilted as the exact guard holds them,
and that the owner's lines clear them in the planner's world (fix plan Track S).

Run from the repository root with the repository's own interpreter; the sidecar is spawned with the cuRobo
environment's, as a cell spawns it::

    python scripts/curobo/probe_support_solids.py
    python scripts/curobo/probe_support_solids.py --curobo-python <cuRobo env python> --coal-prefix <coal env>
    python scripts/curobo/probe_support_solids.py --cpu-replica <cuRobo content root>

Why it exists. The camera world now finds what the parts stand on (``support_surfaces``) and holds each surface as
solids that follow it as the camera read it: the owner's foam mat reads about a degree off level, so its solids tilt.
The planner gets each solid's whole turn as the pose's quaternion (``world.planner_cuboid(..., rotation=...)``) and the
exact guard a turned coal box; the CPU suite holds the wire and the coal box. Only the GPU kernel can say that cuRobo's
world term measures the box it was handed, tilted, and not an upright one, and that the owner's lines clear the solids
in the planner's world as they clear them in the guard's.

The cell. The owner's camera world and workspace (``tests/_cell_2026_10_01.owner_robot``: the cell's 5 mm distances,
the bench declared under the work area, supports on with the 2 mm allowance), with the hand W's probes start: the Hand-E
on a 20 mm plate, approach +Z and closing +X on the flange, planner_margin_mm 4.0, whose retract row and evidence are
committed, with the guard margin that evidence was measured at (10 mm; the owner's cell keeps 5, which no committed
evidence measured). The owner's own hand, on a 23 mm plate with its PolyScope tool frame, has neither yet, and the
driver's start refuses it. Each of the owner's poses is taken over by putting the probe's TCP frame on the owner's, so
the jaws close the same way, moved 1.75 mm further along the approach, so the probe's fingertips stand where the
owner's did (that plate is 3 mm thicker and that TCP 1.25 mm further out). The probe's wrist and flange then stand
3 mm nearer the work than the owner's.

What it does, in order:

  1. starts the planner through the driver's own start, which refuses a sidecar whose composed_sha256 is not the one the
     committed evidence names;
  2. tilted boxes: at three of the owner's configurations, an 800 x 600 x 40 mm box tilted 1, 3 and 5 degrees under the
     hand, lowered until the robot's own spheres keep 10 mm from it (cuRobo's spheres, placed by the CPU replica's
     forward kinematics of the composed robot); the clearance at which the planner's world term turns from clear to
     refused, found by bisection, must lie within 0.5 mm of that distance, and the same box upright (turned about z
     alone) must turn at its own, larger distance;
  3. R8's line in (robot.log:503, 80 mm in 2 mm steps) in the world the cell builds of each recorded look, a 40 mm
     cylinder drawn at the grasp and held out as the target: every sample clear of the solids in the planner's world;
  4. the carried lift's first sample, which is the grasp itself and which no admission sets aside: over bare spots of
     each look, a 30 and a 42 mm part drawn in and held out, the lowest vertical grasp the exact guard admits, from
     SFE's lowest up in 1 mm steps (the owner's TCP 25.9 mm over the support SFE takes, the higher of the mat's local
     reading and the support model's under the part's foot): no admitted grasp's spheres reach into a solid;
  5. the 14:58 scene drawn into P1 (the green cylinder between two 40 mm cubes, the photo's layout): the owner's 10 mm
     standoff (Test_problem.txt:56) and the grasp 10 mm further, both clear of the solids in the planner's world.

Steps 3 to 5 judge each configuration twice, at a ladder of clearances (0 to 20 mm), and report the largest it clears:
against the world's support and bench solids alone, which is what they expect, and against the whole world. The
camera's other boxes are not S's: where only they refuse a straight line, W's admission leaves them to the exact guard,
which holds the same boxes. The probe prints one line per step and writes ``logs/curobo/support_solids_probe.json``.
Exit code 0 when every expectation held, 1 otherwise. The planner is stopped on every way out. On the GPU it spawns a
sidecar of its own, so stop the console, the API and every other planner first: one sidecar on the GPU at a time. The
probe's arm carries no wrist camera, so the robobrain.eye housing is not in it.

``--cpu-replica <cuRobo content root>`` judges with the CPU replica of the sidecar that ``probe_band_admission.py``
defines, every line tagged ``[replica]``: it proves the geometry and the wire, not the GPU kernel. Step 2 places the
spheres with the replica in either mode, from ``--content-root`` (default ``WILLY_PROBE_CUROBO_CONTENT``, else
``ext_deps/curobo/curobo/content``), which must hold the descriptor and the URDF.
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

#: The owner's hand: the Hand-E on a 23 mm plate under its PolyScope tool frame, 157 mm out and turned -90 deg about Z.
_OWNER_PLATE_MM = 23.0
_OWNER_TOOL = ((0.0, 0.0, 157.0), (0.0, 0.0, -0.7071068, 0.7071068))
#: The probe's hand: W's committed composition, the Hand-E on a 20 mm plate, approach +Z and closing +X.
_PROBE_PLATE_MM = 20.0
_PROBE_TOOL = ((0.0, 0.0, 155.75), (0.0, 0.0, 0.0, 1.0))
#: How much further along the approach the probe's TCP stands than the owner's, for its fingertips to stand where
#: the owner's did.
_TIP_SHIFT_MM = (_OWNER_PLATE_MM - _PROBE_PLATE_MM) - (_OWNER_TOOL[0][2] - _PROBE_TOOL[0][2])
#: R8's refused sample (robot.log:503): its joints, its index in a line of 46 samples, the line 80 mm long.
_R8_DEG = (-99.3, -95.7, -127.4, -46.9, 89.7, -281.4)
_R8_SAMPLE, _R8_SAMPLES, _R8_LINE_MM = 43, 46, 80.0
#: The 14:58 standoff, goal 1 of 8 (Test_problem.txt:56), radians.
_GOAL1_RAD = (-1.2675, -1.5194, -2.3572, -0.8358, 1.5708, -1.2576)
#: The photo's layout (2026-10-01): the green cylinder's axis, radius and top; the two 40 mm cubes and their turns.
_CYLINDER = ((-4.5, -534.5), 20.0, 100.5)
_CUBES = (((-110.2, -531.2), 1.0), ((78.2, -531.6), 6.0))
#: Spots on the recorded mat for step 4; one with a part standing near it is reported and left out.
_SPOTS = ((-380.0, -800.0), (-380.0, -640.0), (-340.0, -720.0), (-340.0, -560.0), (-300.0, -850.0), (-300.0, -760.0))
#: SFE's lowest vertical grasp: the owner's TCP this far over the support under it (the fingertips 12.3 mm
#: below it).
_SFE_TCP_OVER_SUPPORT_MM = 25.9
_OWNER_TIP_BELOW_TCP_MM = 12.3
#: Step 2's box, millimetres, and how far the robot's spheres keep from it.
_BOX_MM = (800.0, 600.0, 40.0)
_BOX_DISTANCE_MM = 10.0
#: The clearances steps 3 to 5 ask the planner's world term at, millimetres.
_LADDER_MM = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _probe_robot() -> dict[str, Any]:
    """The owner's ``robot`` block with the probe's hand in place of the owner's, and the guard margin its evidence was
    measured at (10 mm against the owner's 5): the planner's world term, which is what this probe asks, reads neither."""
    from tests import _cell_2026_10_01 as cell

    robot = cell.owner_robot()
    robot["gripper"] = {"model": "robotiq_hande",
                        "coupling_plates": [{"name": "adapter", "thickness_mm": _PROBE_PLATE_MM}],
                        "tool_frame": {"source": "willy", "offset_mm": list(_PROBE_TOOL[0]),
                                       "rotation_quat_xyzw": list(_PROBE_TOOL[1])}}
    robot["safety"]["self_collision"]["min_distance_mm"] = 10.0
    return robot


class _Hands:
    """The owner's TCP at given joints, and the probe's joints that put its fingertips where the owner's were."""

    def __init__(self, owner_cell: Any) -> None:
        from src.robot.drivers.ur.tool_frame import tool_frame_matrix

        self.cell = owner_cell
        self.owner_tool = np.asarray(tool_frame_matrix(*_OWNER_TOOL), dtype=np.float64)
        self.shift = np.eye(4)
        self.shift[2, 3] = _TIP_SHIFT_MM

    def owner_tcp(self, joints_rad: Any) -> np.ndarray:
        return np.asarray(self.cell.tcp(joints_rad) @ np.linalg.inv(self.cell.tool) @ self.owner_tool)

    def goal(self, owner_tcp: np.ndarray) -> np.ndarray:
        """The probe's TCP for the owner's: that frame, moved along its approach by :data:`_TIP_SHIFT_MM`."""
        return np.asarray(owner_tcp @ self.shift)

    @staticmethod
    def seed(owner_joints_rad: Any) -> list[float]:
        """The owner's joints with wrist 3 turned back the quarter turn the owner's tool frame has over the probe's."""
        q = [float(v) for v in owner_joints_rad]
        q[5] -= math.pi / 2.0
        return q

    def joints(self, goal: np.ndarray, near_rad: Any) -> list[float]:
        q = np.asarray(self.cell.ik_tcp(goal, near_rad), dtype=np.float64)
        reached = self.cell.tcp(q)
        miss = float(np.linalg.norm(reached[:3, 3] - goal[:3, 3]))
        cosine = (float(np.trace(reached[:3, :3].T @ goal[:3, :3])) - 1.0) / 2.0
        turn = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
        if miss > 0.05 or turn > 0.01:
            raise RuntimeError(f"no joints put the probe's TCP on its goal: {miss:.3f} mm and {turn:.4f} deg off")
        q[5] = math.remainder(float(q[5]), 2.0 * math.pi)
        return [float(v) for v in q]


def _world_ok(planner: Any, configs: "list[list[float]]", clearance_mm: float = 0.0) -> "list[bool]":
    """Per configuration, whether the planner's world term clears it at ``clearance_mm``; its other terms apart."""
    judged = planner.judge_joint_path(configs, clearance_mm=clearance_mm, name_pairs=False)
    refused = {int(row.index): row for row in judged.refused}
    return [index not in refused or bool(refused[index].world_ok) for index in range(len(configs))]


def _ladder(planner: Any, configs: "list[list[float]]") -> "list[float]":
    """Per configuration, the largest clearance of :data:`_LADDER_MM` the planner's world term clears it at, or -1
    where it refuses it at every one."""
    cleared = [-1.0] * len(configs)
    for rung in _LADDER_MM:
        for index, ok in enumerate(_world_ok(planner, configs, rung)):
            if ok:
                cleared[index] = max(cleared[index], rung)
    return cleared


def _planner_mm(planner: Any, joints: "list[float]", high_mm: float = 60.0) -> float:
    """The clearance at which the planner's world term turns from clear to refused, by bisection, millimetres."""
    if not _world_ok(planner, [joints], 0.0)[0]:
        return 0.0
    if _world_ok(planner, [joints], high_mm)[0]:
        return math.inf
    low, high = 0.0, high_mm
    for _ in range(16):
        middle = (low + high) / 2.0
        if _world_ok(planner, [joints], middle)[0]:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _analytic_mm(spheres_m: np.ndarray, box: dict) -> float:
    """The least distance of the spheres (BASE metres, radius last) to a wire box, millimetres: centre to box, less
    the radius, as the planner's world term measures it."""
    from probe_band_admission import _quat_wxyz

    x, y, z, qw, qx, qy, qz = (float(v) for v in box["pose"])
    local = (spheres_m[:, :3] - np.array([x, y, z])) @ _quat_wxyz([qw, qx, qy, qz])
    half = np.asarray(box["dims_m"], dtype=np.float64) / 2.0
    gap = (np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)
           + np.minimum(np.max(np.abs(local) - half, axis=1), 0.0) - spheres_m[:, 3])
    return float(gap.min()) * 1000.0


def _tilted_boxes(probe: "Probe", planner: Any, spheres_base: Any, configurations: "dict[str, list[float]]") -> dict:
    from src.robot.safety.planning.world import planner_cuboid

    declared = [dict(box) for box in planner._world_cuboids]  # noqa: SLF001 (the declared world under every box)
    yaw = math.radians(35.0)
    out: dict[str, Any] = {}
    for (name, joints), tilt in zip(configurations.items(), (1.0, 3.0, 5.0)):
        spheres = spheres_base(joints)
        lowest = spheres[int(np.argmin(spheres[:, 2] - spheres[:, 3]))] * 1000.0
        turn = _rz(yaw) @ _rx(math.radians(tilt))
        # Off to the side, so the top the hand looks down on rises with the tilt: the upright box keeps more.
        x, y = float(lowest[0]) + 150.0, float(lowest[1]) - 100.0

        def boxes(z_mm: float, x: float = x, y: float = y, turn: np.ndarray = turn) -> "tuple[dict, dict]":
            return (planner_cuboid("seen_s00_support0", (x, y, z_mm), _BOX_MM, rotation=turn.reshape(-1).tolist()),
                    planner_cuboid("seen_s00_support0", (x, y, z_mm), _BOX_MM, yaw_rad=yaw))

        low, high = float(lowest[2]) - 2000.0, float(lowest[2])
        for _ in range(60):
            middle = (low + high) / 2.0
            if _analytic_mm(spheres, boxes(middle)[0]) > _BOX_DISTANCE_MM:
                low = middle
            else:
                high = middle
        rows: dict[str, Any] = {}
        for label, box in zip(("tilted", "upright"), boxes(low)):
            analytic = _analytic_mm(spheres, box)
            under = min((_analytic_mm(spheres, other) for other in declared), default=math.inf)
            planner.set_world([box])
            rows[label] = {"analytic_mm": round(analytic, 3), "declared_mm": round(under, 3),
                           "planner_mm": round(_planner_mm(planner, joints), 3), "pose": box["pose"]}
        tilted, upright = rows["tilted"], rows["upright"]
        held = all(abs(row["planner_mm"] - min(row["analytic_mm"], row["declared_mm"])) <= 0.5
                   and row["analytic_mm"] < row["declared_mm"] - 1.0 for row in rows.values())
        apart = upright["analytic_mm"] - tilted["analytic_mm"] > 1.0
        out[name] = {"tilt_deg": tilt, **rows}
        probe.expect(f"a box tilted {tilt:g} deg under {name}", held and apart,
                     f"the planner turns at {tilted['planner_mm']:.2f} mm for the tilted box (the spheres say "
                     f"{tilted['analytic_mm']:.2f}) and at {upright['planner_mm']:.2f} mm for it upright (the spheres "
                     f"say {upright['analytic_mm']:.2f}); the declared world is {tilted['declared_mm']:.1f} mm away")
    return out


def _cloud(look: Any) -> np.ndarray:
    """The look's points, BASE millimetres, every second pixel."""
    depth = look.surface_depth_mm[::2, ::2]
    v, u = np.mgrid[0:look.surface_depth_mm.shape[0]:2, 0:look.surface_depth_mm.shape[1]:2]
    ok = depth > 0.0
    k, to_base = look.intrinsics, look.camera_to_base
    camera = np.column_stack(((u[ok] - k[0, 2]) * depth[ok] / k[0, 0], (v[ok] - k[1, 2]) * depth[ok] / k[1, 1],
                              depth[ok]))
    return np.asarray(camera @ to_base[:3, :3].T + to_base[:3, 3])


def _cylinder_points(x: float, y: float, z0: float, z1: float, radius: float) -> np.ndarray:
    """A segmented cylinder's points: its lid in rings 2 mm apart and its side in rows 2 mm apart."""
    points = []
    for ring in np.arange(0.0, radius + 0.1, 2.0):
        for angle in np.linspace(0.0, 2.0 * math.pi, max(1, int(math.pi * ring)), endpoint=False):
            points.append((x + ring * math.cos(angle), y + ring * math.sin(angle), z1))
    for z in np.arange(z0 + 1.0, z1, 2.0):
        for angle in np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False):
            points.append((x + radius * math.cos(angle), y + radius * math.sin(angle), z))
    return np.asarray(points)


def _world(owner: Any, look: Any, depth: np.ndarray, target: np.ndarray, goal_tcp: np.ndarray, envelope: Any,
           label: str = "part") -> Any:
    """The world the cell builds of ``look`` read as ``depth``, ``target`` held out, the jaws' space at the goal kept."""
    world = owner.world(look, depth_mm=depth)
    world.offer_segmentation(camera="EIH_Cam", target_points_base_mm=target, target_label=label, hold=True)
    flange = goal_tcp @ np.linalg.inv(owner.tool)
    snapshot = world.world_for(self_envelope=envelope, near_point_mm=flange[:3, 3],
                               goal_keep_out=owner.goal_keep_out(goal_tcp))
    if snapshot.perceived is None:
        raise RuntimeError(f"the camera world was refused: {snapshot.reason}")
    return snapshot


def _ladders(probe: "Probe", planner: Any, snapshot: Any, configs: "list[list[float]]", name: str) -> dict:
    """The ladder of each configuration against the world's support and bench solids alone, then against the whole
    world, each registered with the declared world under it."""
    from src.robot.safety.planning.world import merge_planner_worlds

    perceived = snapshot.perceived
    solids = sorted(box.name for box in perceived.boxes if box.kind != "seen")
    if not solids:
        probe.expect(f"{name}: the world holds its solids", False, "the camera world found no support or bench")
    row: dict[str, Any] = {"solids": solids,
                           "supports": "" if perceived.supports is None else perceived.supports.render()}
    for label, cuboids in (("solids", [dict(c) for c in snapshot.cuboids if c["name"] in solids]),
                           ("whole", [dict(c) for c in snapshot.cuboids])):
        sent = merge_planner_worlds(planner._world_cuboids, cuboids)  # noqa: SLF001
        registered = planner.set_world(cuboids)
        if registered != len(sent):
            probe.expect(f"{name}: the {label} world registered", False, f"{registered} of {len(sent)} boxes")
        row[f"{label}_boxes"] = registered
        row[f"{label}_cleared_mm"] = _ladder(planner, configs)
    return row


class Probe:
    def __init__(self) -> None:
        self.failed: list[str] = []
        self.steps: dict[str, Any] = {}

    def expect(self, name: str, ok: bool, said: str) -> None:
        say(f"  {'ok  ' if ok else 'FAIL'} {name}: {said}")
        if not ok:
            self.failed.append(name)


def _r8_line(probe: Probe, owner: Any, hands: _Hands, envelope: Any) -> dict:
    from tests import _cell_2026_10_01 as cell

    r8 = np.radians(_R8_DEG)
    refused_at = hands.owner_tcp(r8)
    approach = refused_at[:3, 2]
    his_grasp = refused_at.copy()
    his_grasp[:3, 3] += approach * (_R8_SAMPLES - 1 - _R8_SAMPLE) * _R8_LINE_MM / (_R8_SAMPLES - 1)
    grasp = hands.goal(his_grasp)
    poses, near = [], hands.seed(r8)
    for k in range(int(round(_R8_LINE_MM / 2.0)) + 1):
        step = grasp.copy()
        step[:3, 3] = grasp[:3, 3] - approach * (_R8_LINE_MM - 2.0 * k)
        near = hands.joints(step, near)
        poses.append(near)
    gx, gy, gz = (float(v) for v in his_grasp[:3, 3])
    lines: dict[str, Any] = {}
    for name in cell.LOOKS:
        look = cell.look(name)
        mat = cell.local_reading(look, gx, gy)
        if not math.isfinite(mat):
            probe.expect(f"R8's line on {name}", False, f"the look reads no mat round the grasp ({gx:.1f}, {gy:.1f})")
            continue
        depth = cell.with_solids(look, cylinders=[(gx, gy, 20.0, mat, gz + 4.0)], seed=sum(map(ord, "R8")))
        snapshot = _world(owner, look, depth, _cylinder_points(gx, gy, mat, gz + 4.0, 20.0), grasp, envelope)
        row = _ladders(probe, _planner_of(owner), snapshot, poses, f"R8 on {name}")
        row.update(samples=len(poses), his_tip_over_mat_mm=round(gz - _OWNER_TIP_BELOW_TCP_MM - mat, 1))
        lines[name] = row
        solid, whole = row["solids_cleared_mm"], row["whole_cleared_mm"]
        probe.expect(f"R8's line on {name}", min(solid) >= 0.0,
                     f"{sum(c >= 0.0 for c in solid)} of {len(poses)} samples clear of the solids in the planner's "
                     f"world, the least at {min(solid):g} mm of the ladder ({len(row['solids'])} solids); against the "
                     f"whole world {sum(c >= 0.0 for c in whole)} clear, the least at {min(whole):g} mm (the camera's "
                     "boxes, which W's admission leaves to the exact guard)")
    return lines


def _lowest_admitted(owner: Any, hands: _Hands, tcp: np.ndarray, support: float, boxes: Any,
                     seed: "list[float]") -> "tuple[float | None, list[float] | None, str]":
    """From SFE's lowest vertical grasp up in 1 mm steps (12 at most), the first the exact guard admits: how far it was
    raised, its joints, and what the guard said of the last one it refused."""
    said = ""
    for raised in range(13):
        tcp[:3, 3] = (tcp[0, 3], tcp[1, 3], support + _SFE_TCP_OVER_SUPPORT_MM + float(raised))
        q = hands.joints(hands.goal(tcp), seed)
        said = owner.refused(boxes, q)
        if not said:
            return float(raised), q, ""
    return None, None, said


def _lifts(probe: Probe, owner: Any, hands: _Hands, envelope: Any) -> dict:
    """At each bare spot, the lowest vertical grasp the exact guard admits from SFE's lowest up, and the planner's world
    term there (the lift's first sample, which no admission sets aside)."""
    from src.robot.safety.planning.live_world import _guard_boxes
    from tests import _cell_2026_10_01 as cell

    if not owner.has_the_engine():
        probe.expect("the lift's first sample", False,
                     "the exact guard has no engine (Coal) on this interpreter, so no grasp can be admitted")
        return {}
    r8 = np.radians(_R8_DEG)
    down = hands.owner_tcp(r8)
    grasps: dict[str, Any] = {}
    left_out: list[str] = []
    for name in cell.LOOKS:
        look = cell.look(name)
        cloud = _cloud(look)
        for x, y in _SPOTS:
            mat = cell.local_reading(look, x, y)
            near_spot = np.hypot(cloud[:, 0] - x, cloud[:, 1] - y) < 70.0
            if not math.isfinite(mat) or int(np.count_nonzero(near_spot & (cloud[:, 2] > mat + 12.0))) > 20:
                left_out.append(f"{name} ({x:g}, {y:g})")
                continue
            for height in (30.0, 42.0):
                key = f"{name} {height:g} mm part at ({x:g}, {y:g})"
                tcp = down.copy()
                tcp[:3, 3] = (x, y, mat + _SFE_TCP_OVER_SUPPORT_MM)
                depth = cell.with_solids(look, cylinders=[(x, y, 20.0, mat, mat + height)], seed=int(height))
                snapshot = _world(owner, look, depth, _cylinder_points(x, y, mat, mat + height, 20.0), hands.goal(tcp),
                                  envelope)
                model = snapshot.perceived.supports
                under = None if model is None else model.height_under(
                    _cylinder_points(x, y, mat, mat + 2.0, 20.0)[:, :2])
                support = mat if under is None else max(mat, float(under))
                boxes = _guard_boxes(snapshot.perceived)
                raised, q, said = _lowest_admitted(owner, hands, tcp, support, boxes, hands.seed(r8))
                if raised is None or q is None:
                    grasps[key] = {"mat_mm": round(mat, 1), "support_mm": round(support, 1), "admitted": False,
                                   "guard": said[:240]}
                    continue
                row = _ladders(probe, _planner_of(owner), snapshot, [q], key)
                grasps[key] = {"mat_mm": round(mat, 1), "support_mm": round(support, 1), "admitted": True,
                               "raised_mm": raised, "guard_least_mm": round(owner.distance_mm(q, boxes), 2),
                               "solids_cleared_mm": row["solids_cleared_mm"][0],
                               "whole_cleared_mm": row["whole_cleared_mm"][0], "solids": row["solids"]}
    admitted = [row for row in grasps.values() if row["admitted"]]
    solid = [row["solids_cleared_mm"] for row in admitted]
    whole = [row["whole_cleared_mm"] for row in admitted]
    raised_by = ", ".join(sorted({f"{row['raised_mm']:g}" for row in admitted}, key=float)) or "-"
    probe.expect("no admitted grasp reaches into a solid", len(admitted) >= 12 and min(solid, default=-1.0) >= 0.0,
                 f"{len(admitted)} of {len(grasps)} grasps admitted by the guard (raised over SFE's lowest by "
                 f"{raised_by} mm); {sum(c >= 0.0 for c in solid)} of them clear of the solids in the planner's world, "
                 f"the least at {min(solid, default=-1.0):g} mm of the ladder; against the whole world "
                 f"{sum(c >= 0.0 for c in whole)} clear; spots left out for a part near: {left_out or 'none'}")
    return {"grasps": grasps, "left_out_for_a_part_near": left_out}


def _standoff_14_58(probe: Probe, owner: Any, hands: _Hands, envelope: Any) -> dict:
    from tests import _cell_2026_10_01 as cell

    look = cell.look("P1")
    (cx, cy), radius, top = _CYLINDER
    mat = cell.local_reading(look, cx, cy)
    cubes = []
    for (x, y), yaw_deg in _CUBES:
        under = cell.local_reading(look, x, y)
        cubes.append(((x, y, (under if math.isfinite(under) else mat) + 20.0), (40.0, 40.0, 40.0),
                      _rz(math.radians(yaw_deg))))
    depth = cell.with_solids(look, cylinders=[(cx, cy, radius, mat, top)], boxes=cubes, seed=1)
    his_standoff = hands.owner_tcp(_GOAL1_RAD)
    his_grasp = his_standoff.copy()
    his_grasp[:3, 3] += his_standoff[:3, 2] * 10.0
    standoff_goal, grasp_goal = hands.goal(his_standoff), hands.goal(his_grasp)
    snapshot = _world(owner, look, depth, _cylinder_points(cx, cy, mat, top, radius), standoff_goal, envelope,
                      label="Einen gruenen Zylinder.")
    standoff = hands.joints(standoff_goal, hands.seed(_GOAL1_RAD))
    row = _ladders(probe, _planner_of(owner), snapshot, [standoff, hands.joints(grasp_goal, standoff)],
                   "the 14:58 scene")
    row["mat_mm"] = round(mat, 1)
    solid, whole = row["solids_cleared_mm"], row["whole_cleared_mm"]
    probe.expect("the 14:58 standoff and grasp", min(solid) >= 0.0,
                 f"the solids in the planner's world clear the standoff at {solid[0]:g} mm and the grasp at "
                 f"{solid[1]:g} mm of the ladder ({len(row['solids'])} solids); the whole world at {whole[0]:g} and "
                 f"{whole[1]:g} mm (the camera's boxes, which W's admission leaves to the exact guard)")
    return row


def _planner_of(owner: Any) -> Any:
    """The planner glue the arm started."""
    return owner.arm._curobo_ur  # noqa: SLF001


def _steps(probe: Probe, owner: Any, planner: Any, spheres_base: Any) -> None:
    from tests import _cell_2026_10_01 as cell

    hands = _Hands(owner)
    envelope = owner.envelope(np.radians(cell.LOOK_0_DEG))
    say("2. tilted boxes against the spheres' own distance:")
    probe.steps["tilted"] = _tilted_boxes(probe, planner, spheres_base, {
        "LOOK[0]": [math.radians(v) for v in cell.LOOK_0_DEG],
        "R8's refused sample": [math.radians(v) for v in _R8_DEG],
        "the 14:58 standoff": list(_GOAL1_RAD)})
    say("3. R8's line in over the owner's recorded mat:")
    probe.steps["r8"] = _r8_line(probe, owner, hands, envelope)
    say("4. the carried lift's first sample at the lowest vertical grasp the guard admits over bare mat:")
    probe.steps["lifts"] = _lifts(probe, owner, hands, envelope)
    say("5. the 14:58 scene drawn into P1, the 10 mm standoff and the grasp:")
    probe.steps["14:58"] = _standoff_14_58(probe, owner, hands, envelope)


def run(args: argparse.Namespace) -> int:
    if args.cpu_replica:
        tag_replica_output()
    if args.curobo_python:
        os.environ["WILLY_CUROBO_PYTHON"] = args.curobo_python
    if args.coal_prefix:
        os.environ["WILLY_COAL_PREFIX"] = args.coal_prefix
    if args.stderr_log:
        os.environ["WILLY_CUROBO_STDERR"] = args.stderr_log
    from tests import _cell_2026_10_01 as cell

    owner = cell.OwnerCell(_probe_robot())
    arm = owner.arm
    replica: "CpuReplicaClient | None" = None
    if args.cpu_replica:
        replica = CpuReplicaClient(arm, Path(args.cpu_replica))
        setattr(arm, "_curobo_client_factory", lambda: replica)  # noqa: B010 (the probe's own stand-in, by name)
    content = Path(args.content_root or os.environ.get("WILLY_PROBE_CUROBO_CONTENT", "")
                   or REPO / "ext_deps" / "curobo" / "curobo" / "content")
    placer = CpuReplicaClient(arm, content)
    placer.start()

    def spheres_base(joints: "list[float]") -> np.ndarray:
        """cuRobo's spheres at ``joints`` in the controller's base, metres: the planner's frame is half a turn about Z."""
        spheres = np.array(placer._spheres(list(joints))[0], dtype=np.float64)  # noqa: SLF001
        spheres[:, :2] *= -1.0
        return spheres[spheres[:, 3] > 0.0]

    arm._conn = MagicMock()  # noqa: SLF001
    arm._conn.is_connected = True  # noqa: SLF001
    arm._conn.get_joint_positions.return_value = [math.radians(v) for v in cell.LOOK_0_DEG]  # noqa: SLF001
    probe = Probe()
    out: dict[str, Any] = {
        "planner": "CPU REPLICA, not the GPU sidecar" if replica is not None else "GPU sidecar",
        "composition": "UR10, Hand-E on a 20 mm plate, approach +Z closing +X, margin 4 mm (the committed one W's "
                       "probes start); the owner's poses taken over with the fingertips where the owner's were "
                       f"({_TIP_SHIFT_MM:.2f} mm along the approach)",
    }
    say(f"planner under test: {out['planner']}")
    started = time.perf_counter()
    try:
        identity = arm.start_planner()
        out["start_s"] = round(time.perf_counter() - started, 1)
        out["composed_sha256"] = identity.composed_sha256
        say(f"1. planner started in {out['start_s']} s, composed_sha256 {identity.composed_sha256}")
        _steps(probe, owner, arm._curobo_ur, spheres_base)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 (reported with its trace, and the planner still stops)
        probe.failed.append(f"{type(exc).__name__}: {exc}")
        out["error"] = traceback.format_exc(limit=12)
        say(out["error"])
    finally:
        arm.stop_planner()
    out["steps"] = probe.steps
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
                             "under this cuRobo content root. Every line it prints then starts with [replica]")
    parser.add_argument("--content-root", default="", metavar="CONTENT_ROOT",
                        help="the cuRobo content root whose descriptor and URDF place the spheres step 2 measures with "
                             "(default WILLY_PROBE_CUROBO_CONTENT, else ext_deps/curobo/curobo/content)")
    parser.add_argument("--out", default=str(REPO / "logs" / "curobo" / "support_solids_probe.json"))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
