"""The push's simulation gate (fix plan, Track P): a cell's own tree, URSim CB3, the toggle hand, the owner's mat.

    .venv/Scripts/python.exe scripts/ursim/probe_push_on_the_mat.py --tree D:/cells/mine/config --work D:/runs/p1 \\
        --profile monday,ursim_mat --calibration D:/cells/mine/calibration/eih_EIH_Cam.json --looks first
    .venv/Scripts/python.exe scripts/ursim/probe_push_on_the_mat.py ... --offline      # step 0 alone, nothing connects

It copies the tree into ``--work`` and adds one layer of its own, ``ursim_mat``: ``robot.ur.ip`` 127.0.0.1 and nothing
else (``--calibration`` adds a camera layer that points the primary rig at that artifact, the rig otherwise as the tree
has it). A chain that does not name the layer, or one whose address resolves to anything but 127.0.0.1, is refused
before anything connects. Everything else is the tree's: the arm, the toggle hand on its output, the tool frame, the
looks, the guard's distances, the planner and its world.

URSim has no camera and moves no part, so ``_mat_scene.py`` stands in for the camera and for the parts, and for nothing
else: one recorded look of the owner's mat (Track K's crop of P1) with the scene rendered in, from wherever the TCP
stands at each grab, through the rig's own calibration. The detector and the segmenter ground the target on that
render; the calculator, the camera world, S's support model, the blocker choice, ``plan_push``, ``push_motion``, the
exact guard, the planner, the toggle and the budgets are the cell's own. After a push the part stands where the plan
put it; a part the jaws close on is carried and set down where they open.

Step 0, offline, before anything connects: the push scene rendered from each look, S's model, and ``plan_push`` as the
pick asks it. It reports and does not gate. Then the scenarios, on URSim, each a whole pick of the cell's service:

* ``refusal`` (P-2): the cylinder beside a 40 mm cube: no motion toward the part, ``ALL_COLLIDED`` and the push's
  refusal in the record, no change of the tool output.
* ``blocker`` (R): a neighbour gripped, set down on a free spot the camera saw, then the cylinder gripped.
* ``push`` (P-1, items 1-4, 7, 8), ``cancel`` (item 5: a stop asked for while the push leg runs) and ``pstop`` (item 6:
  a protective stop while the push leg runs): the cylinder beside a 25 mm cube, which nothing can grip.

``--scene NAME=NEIGHBOURS@GAP@X,Y`` places a scenario's cylinder and its neighbours (``name[:side[:gap]],...``);
``NAME#n`` runs a scenario again on a scene of its own. ``--looks first`` hands every pick LOOK[0] alone, ``all`` (the
default) the tree's looks, as example 12 does.

``--push-stand-in`` is the plan's stand-in for P-1, said in the result wherever it was used: in ``push``, ``cancel`` and
``pstop`` the calculator answers the part ``ALL_COLLIDED`` until a push ran, handing on its own metadata (Track A's
arrays), and ``plan_push``, where it refuses on the pick's own inputs, is asked again with the part's whole surface and
the mat round it, what cameras all round would see. Every other input, the plan's execution, the guard, the world, the
toggle and the stops stay the cell's own. Without it a push plans only where the cell's own looks see the part's foot
and the table all round its landing.

Tool DO0 is read on a second RTDE connection the probe opens itself; every change of it is counted. The JSON result is
``<work>/p_result.json``. Exit codes: 0 every item held, 1 an item did not (each named), 2 refused before anything
connected (a chain, an address, a work folder, a tree, a controller not up).

Run it with the cuRobo sidecar's environment (``gpu_aurora/run_with_aurora.py <this script> ...`` on this box) and
from a tree whose planner evidence holds the cell's combination.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import sys
import threading
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence
from unittest import mock

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import _mat_scene as ms  # noqa: E402

from src.config.loader import load_robot_config  # noqa: E402
from src.config.tree import load_tree  # noqa: E402
from src.robot.drivers.ur.arm import URRobotArm  # noqa: E402

#: The probe's own layer, and the only address it lets the chain resolve to.
URSIM_LAYER = "ursim_mat"
URSIM_ADDRESS = "127.0.0.1"
_LAYER_TEXT = (
    "# The layer scripts/ursim/probe_push_on_the_mat.py adds to the copy of a cell's tree: the controller is URSim on\n"
    "# this box, and nothing else changes.\n"
    "robot:\n"
    "  ur:\n"
    f"    ip: \"{URSIM_ADDRESS}\"\n"
)
_OK, _FAILED, _REFUSED = 0, 1, 2

#: The owner's LOOK[0] (example 12), where P1 was recorded, in degrees.
LOOK0_DEG = (-87.50, -64.51, -85.79, -129.82, 90.27, -108.84)
#: Track K's crop of P1, the first Zollstock pick, depth only.
DEFAULT_CROP = Path(_REPO_ROOT) / "tests" / "data" / "cell_2026_10_01" / "p1_pick-63e1b3dac29f.npz"
#: Where the scenes stand: the cylinder on the mat where the pile was, its neighbour off its -x side. The plan's
#: (-150, -650) lies 110 mm from the mat's near edge, too near for the landing box a 50 mm push asks; at y -760 the real
#: chain answers the refusal and the forced push as the plan's items ask.
CYLINDER_XY = (-150.0, -760.0)
CYLINDER_MM = (20.0, 50.0)
PROMPT = "green cylinder"
TARGET = "cylinder"
#: The neighbours of the scenarios: a cube nothing can grip (under the Hand-E's least height), one it can, a block
#: wider than the hand opens.
NEIGHBOURS = {"cube25": (25.0, 25.0, 25.0), "cube40": (40.0, 40.0, 40.0), "block60": (60.0, 60.0, 40.0)}
SCENARIO_NEIGHBOUR = {"refusal": "cube40", "blocker": "cube40", "push": "cube25", "cancel": "cube25", "pstop": "cube25"}
DEFAULT_SCENARIOS = ("refusal", "blocker", "push", "cancel", "pstop")
#: The yellow bin in P1 (BASE x, y): its walls stand from x 90 on.
BIN_X_MIN_MM = 90.0
#: The push leg runs at 25 mm/s (push_motion's cap); a moveL at this speed is the push leg.
PUSH_LEG_SPEED_M_S = 0.025
#: How long into the push leg a stop is asked for, seconds (the leg is 60 mm at 25 mm/s).
STOP_INTO_THE_LEG_S = 0.8


# ---------------------------------------------------------------------------------------------------------------------
# The copy, its layer, and the refusals before anything connects
# ---------------------------------------------------------------------------------------------------------------------


def copy_the_tree(tree: "str | Path", work: "str | Path", *, calibration: "str | Path | None" = None) -> Path:
    """Copy ``tree`` (a config folder) to ``<work>/config`` and write the probe's layer over whatever the tree held
    under its name; the tree itself is never written. ``calibration`` adds the camera layer (:func:`_camera_layer`)."""
    copied = Path(work) / "config"
    shutil.copytree(Path(tree), copied)
    layer = copied / "robot" / f"robot.{URSIM_LAYER}.yaml"
    layer.parent.mkdir(parents=True, exist_ok=True)
    layer.write_text(_LAYER_TEXT, encoding="utf-8")
    if calibration is not None:
        _camera_layer(copied, Path(calibration))
    return copied


def _camera_layer(copied: Path, calibration: Path) -> None:
    """``camera/cam.<layer>.yaml``: the tree's rigs as they stand, the primary's artifact at ``calibration``. A list in a
    layer replaces the base list whole, so every rig is restated as the tree has it."""
    base = yaml.safe_load((copied / "camera" / "cam.yaml").read_text(encoding="utf-8")) or {}
    cameras = base.get("cameras") or {}
    primary = cameras.get("primary_rig_id")
    rigs = cameras.get("rigs") or []
    for rig in rigs:
        if rig.get("rig_id") == primary and isinstance(rig.get("extrinsics"), dict):
            rig["extrinsics"]["artifact_path"] = calibration.resolve().as_posix()
    text = ("# The camera layer of scripts/ursim/probe_push_on_the_mat.py: the primary rig's calibration artifact\n"
            "# where this box holds it; every rig otherwise as the tree has it.\n")
    text += yaml.safe_dump({"cameras": {"rigs": rigs}}, sort_keys=False)
    (copied / "camera" / f"cam.{URSIM_LAYER}.yaml").write_text(text, encoding="utf-8")


def _layers(chain: str) -> list[str]:
    return [name.strip() for name in str(chain).split(",") if name.strip()]


def refusal_of(copied: "str | Path", chain: str) -> Optional[str]:
    """Why the copied tree under ``chain`` may not reach a controller, or ``None``: the chain names the probe's layer,
    loads, describes a UR cell, and its controller address is URSim's."""
    if URSIM_LAYER not in _layers(chain):
        return (f"the profile chain {chain!r} does not name the {URSIM_LAYER!r} layer this probe writes, so the "
                f"controller it would reach is whatever the tree names; add {URSIM_LAYER!r} to the chain")
    try:
        robot = load_robot_config(Path(copied), profile=chain)
    except Exception as exc:  # noqa: BLE001 (any tree that does not load is refused here, with what it said)
        return f"the copied tree does not load under {chain!r}: {type(exc).__name__}: {exc}"
    vendor = str(getattr(robot.vendor, "value", robot.vendor))
    if vendor != "ur":
        return f"the copied tree under {chain!r} describes a {vendor!r} cell; this probe drives a UR controller only"
    address = str(robot.ur.ip)
    if address != URSIM_ADDRESS:
        return (f"under {chain!r} the controller address resolves to {address}, not {URSIM_ADDRESS}: a layer after "
                f"{URSIM_LAYER!r} names a controller again, and this probe drives URSim on this box only")
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------------------------------


def _ur_pose_matrix(pose: Sequence[float]) -> np.ndarray:
    """``[x, y, z, rx, ry, rz]`` (m, axis-angle) as a 4x4 in mm."""
    rot = np.asarray(pose[3:6], dtype=np.float64)
    angle = float(np.linalg.norm(rot))
    m = np.eye(4)
    if angle > 1e-12:
        k = rot / angle
        kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        m[:3, :3] = np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)
    m[:3, 3] = np.asarray(pose[:3], dtype=np.float64) * 1000.0
    return m


def _nominal_tcp(robot: Any, joints_deg: Sequence[float]) -> np.ndarray:
    """The TCP at ``joints_deg`` by the bundled DH chain and the tree's declared tool frame, 4x4 BASE mm."""
    from src.geometry.quaternion import to_rotation_matrix  # noqa: PLC0415
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: PLC0415

    chain = ur_link_transforms_mm(str(robot.ur.model), np.radians(np.asarray(joints_deg, dtype=np.float64)))
    if chain is None:
        raise ValueError(f"no bundled DH chain for {robot.ur.model!r}, so no camera can be placed at a look")
    flange = np.asarray(chain[-1], dtype=np.float64)
    tool = np.eye(4)
    tool[:3, :3] = to_rotation_matrix(np.asarray(robot.gripper.tool_frame.rotation_quat_xyzw, dtype=np.float64))
    tool[:3, 3] = np.asarray(robot.gripper.tool_frame.offset_mm, dtype=np.float64)
    return flange @ tool


def _primary_calibration(tree: Any) -> Any:
    from src.calibration.rig_calibration import RigCalibration  # noqa: PLC0415

    section = tree.app_config.camera.cameras
    rig = next(r for r in section.rigs if r.rig_id == section.primary_rig_id)
    return RigCalibration.from_config(rig.rig_id, getattr(rig, "extrinsics", None))


SIDES = ("-x", "+x", "-y", "+y")


def neighbours_of(spec: str) -> list[tuple[str, str, Optional[float]]]:
    """``name[:side[:gap]],...`` as (name, side, gap) triples: where the neighbour stands off the cylinder (-x unless
    said) and its own gap in mm (the scene's unless said). ``ValueError`` names what is not a known neighbour, side or
    number."""
    out = []
    for item in spec.split(","):
        fields = item.strip().split(":")
        name = fields[0]
        side = fields[1] if len(fields) > 1 and fields[1] else "-x"
        gap = fields[2] if len(fields) > 2 else ""
        if name not in NEIGHBOURS or side not in SIDES or len(fields) > 3:
            raise ValueError(f"{item!r}: neighbours are {sorted(NEIGHBOURS)}, sides {list(SIDES)}")
        out.append((name, side, float(gap) if gap else None))
    return out


def place_scene(scene: ms.Scene, neighbour: str, gap_mm: float, *, at_xy: Sequence[float] = CYLINDER_XY
                ) -> dict[str, Any]:
    """The cylinder at ``at_xy`` on the mat's reading, each neighbour of ``neighbour`` (``name[:side],...``)
    ``gap_mm`` off the side it names (its -x side unless said)."""
    radius, height = CYLINDER_MM
    foot, normal = scene.foot_on_mat(at_xy)
    parts = [ms.cylinder(TARGET, foot, normal, radius_mm=radius, height_mm=height)]
    feet = []
    for index, (name, side, own_gap) in enumerate(neighbours_of(neighbour)):
        gap = gap_mm if own_gap is None else own_gap
        sx, sy, sz = NEIGHBOURS[name]
        sign = -1.0 if side[0] == "-" else 1.0
        if side[1] == "x":
            centre = (float(at_xy[0]) + sign * (radius + gap + sx / 2.0), float(at_xy[1]))
        else:
            centre = (float(at_xy[0]), float(at_xy[1]) + sign * (radius + gap + sy / 2.0))
        nfoot, nnormal = scene.foot_on_mat(centre)
        parts.append(ms.box(name if index == 0 else f"{name}_{index}", nfoot, nnormal, size_mm=(sx, sy, sz)))
        feet.append([round(float(v), 1) for v in nfoot])
    scene.set_parts(parts, target=TARGET)
    return {"target": TARGET, "target_foot_mm": [round(float(v), 1) for v in foot], "neighbour": neighbour,
            "neighbour_feet_mm": feet, "gap_mm": gap_mm, "mat_tilt_deg": round(scene.static.mat_tilt_deg, 2)}


# ---------------------------------------------------------------------------------------------------------------------
# Step 0: the push scene offline, S's model and plan_push as the pick asks it
# ---------------------------------------------------------------------------------------------------------------------


class _NoDepth:
    def grab_surface_depth(self) -> None:
        return None


def _looks_deg(robot: Any, which: str) -> list[tuple[float, ...]]:
    """The looks a pick visits: the tree's ``look_joint_positions_deg`` (``all``), or LOOK[0] alone (``first``)."""
    looks = [tuple(float(v) for v in look) for look in (getattr(robot, "look_joint_positions_deg", None) or ())]
    if not looks:
        looks = [LOOK0_DEG]
    return looks if which == "all" else looks[:1]


def offline_step(tree: Any, crop: Path, *, push_mm: float, gaps: Sequence[float], at_xy: Sequence[float],
                 looks: str = "all") -> dict[str, Any]:
    """Render the push scene from each look the pick visits as the cell's camera stands there (the bundled DH chain, the
    declared tool frame, the rig's calibration), find the supports over all of them, and ask ``plan_push`` with the
    cell's push inputs and the table every look saw, gap by gap until one plans. Nothing connects."""
    from src.robot.execution.autonomous_grasp.service import _operator_box  # noqa: PLC0415
    from src.robot.execution.camera_world_wiring import _live_planner_world  # noqa: PLC0415
    from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator  # noqa: PLC0415
    from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415
    from src.robot.grasping.recovery.push_gate import PushCell  # noqa: PLC0415
    from src.robot.grasping.recovery.push_planner import PushPlan, neighbour_evidence, plan_push  # noqa: PLC0415
    from src.robot.core.keep_out import KeepOutBox  # noqa: PLC0415
    from src.robot.safety.planning.live_world import CameraView  # noqa: PLC0415
    from src.robot.safety.planning.perceived import DepthView  # noqa: PLC0415
    from src.robot.safety.planning.self_envelope import ROBOT_BASES  # noqa: PLC0415
    from src.robot.safety.planning.support_surfaces import find_supports  # noqa: PLC0415

    robot = tree.robot
    static = ms.StaticScene.from_crop(crop)
    calibration = _primary_calibration(tree)
    cam_to_tool = np.asarray(calibration.camera_to_tool().to_matrix(), dtype=np.float64)
    poses = [(look, _nominal_tcp(robot, look)) for look in _looks_deg(robot, looks)]
    world = _live_planner_world(robot, [CameraView(name=calibration.rig_id, depth_source=_NoDepth(),  # type: ignore[arg-type]
                                                   camera_to_tool=cam_to_tool)])
    cell = PushCell.from_robot_config(robot)
    if not isinstance(cell, PushCell):
        return {"planned": False, "refused": f"the cell's push inputs: {cell.sentence}"}
    operator_box = _operator_box(robot.grasping.recovery.fixture)
    if isinstance(operator_box, str):
        return {"planned": False, "refused": f"the fixture box: {operator_box}"}
    k = static.intrinsics
    out: dict[str, Any] = {"looks": [list(look) for look, _ in poses],
                           "cameras_mm": [[round(float(v), 1) for v in (tcp @ cam_to_tool)[:3, 3]] for _, tcp in poses],
                           "mat_plane": [round(float(v), 5) for v in static.mat_plane],
                           "mat_rms_mm": round(static.mat_rms_mm, 2), "mat_tilt_deg": round(static.mat_tilt_deg, 2),
                           "pile_points_cut": static.pile_points_cut, "push_mm": push_mm, "at_mm": list(at_xy),
                           "gaps": []}
    for gap in gaps:
        scene = ms.Scene(static)
        placed = place_scene(scene, "cube25", gap, at_xy=at_xy)
        views, targets, neighbours = [], [], []
        for index, (_, tcp) in enumerate(poses):
            cam = tcp @ cam_to_tool
            rendered = scene.render(cam, tcp)
            depth = rendered.depth_mm.astype(np.float64)
            none = np.zeros(depth.shape, dtype=bool)
            targets.append(to_base_mm(rendered.masks.get(TARGET, none), depth, k, cam))
            neighbours.append(to_base_mm(rendered.masks.get("cube25", none), depth, k, cam))
            views.append(DepthView(surface_depth_mm=depth, intrinsics=k, camera_to_base=cam, name=f"look{index}"))
        target, neighbour = np.vstack(targets), np.vstack(neighbours)
        low, high = target.min(axis=0), target.max(axis=0)
        place = np.eye(4)
        place[:3, 3] = (low + high) / 2.0
        keep_out = (KeepOutBox.from_matrix("part_0", place, (high - low) / 2.0 + float(world.tuning.margin_mm)),)
        model = find_supports(views, limits=world.limits, tuning=world.tuning,
                              base=ROBOT_BASES.get(str(robot.ur.model).lower()), keep_out=keep_out)
        surface = model.surface_under(target[:, :2])
        row: dict[str, Any] = {**placed, "target_points": int(target.shape[0]),
                               "neighbour_points": int(neighbour.shape[0]),
                               "points_per_look": [int(t.shape[0]) for t in targets],
                               "supports": [s.render() for s in model.surfaces]}
        if surface is None or surface.offset_mm is None:
            row["refused"] = "no support surface under the part"
            out["gaps"].append(row)
            continue
        normal = np.asarray(surface.normal, dtype=np.float64)
        offset = float(surface.offset_mm)
        table = model.table_points(views, target[:, :2])
        gate = SimpleNamespace(cell=cell, distance_mm=push_mm, operator_box=operator_box)
        floor = BinPickingOrchestrator._finger_floor_mm(model, target, normal, offset, gate)  # type: ignore[arg-type]
        evidence = neighbour_evidence(target_points_mm=target, neighbour_points_mm=neighbour,
                                      support_normal=normal, support_offset_mm=offset)
        floor_kwargs: dict[str, Any] = {} if floor is None else {"finger_floor_mm": floor}
        plan = plan_push(target_points_mm=target, neighbour_points_mm=neighbour, support_normal=normal,
                         support_offset_mm=offset, workspace=cell.workspace, hand=cell.hand, table_points_mm=table,
                         push_distance_mm=push_mm, container_interior=cell.container_interior,
                         operator_box=operator_box, hand_clearance_mm=cell.hand_clearance_mm,
                         beside_part_clearance_mm=cell.beside_part_clearance_mm, beside_part_mm=cell.beside_part_mm,
                         **floor_kwargs)
        row.update({"surface": surface.render(), "surface_tilt_deg": round(float(surface.tilt_deg), 2),
                    "support_offset_mm": round(offset, 2), "table_points": int(np.asarray(table).shape[0]),
                    "finger_floor_mm": None if floor is None else round(float(floor), 2),
                    "evidence_gap_mm": round(float(evidence.nearest_gap_mm), 1), "evidence_found": bool(evidence.found)})
        if isinstance(plan, PushPlan):
            row["plan"] = plan.to_dict()
            row["planned"] = True
            out["gaps"].append(row)
            out["planned"] = True
            out["gap_mm"] = gap
            return out
        row["planned"] = False
        row["refusal"] = plan.to_dict()
        out["gaps"].append(row)
    out["planned"] = False
    return out


# ---------------------------------------------------------------------------------------------------------------------
# What reached the controller, what the arm was asked, what the guard judged
# ---------------------------------------------------------------------------------------------------------------------


class ToolOutput:
    """The cell's toggle output as the controller reports it, read on a second RTDE connection this probe opens, and
    the jaws it drives: every change of it moves them (the owner's valve, 2026-09-28). The jaws start open, as the
    person says at connect. Each change is handed to the scene with the TCP it happened at."""

    #: Bit of each port's pin 0 in ``getActualDigitalOutputBits``: standard 0-7, configurable 8-15, tool 16-17.
    PORT_BIT0 = {"standard": 0, "configurable": 8, "tool": 16}

    def __init__(self, host: str, port: str, pin: int, scene: ms.Scene) -> None:
        import rtde_receive  # noqa: PLC0415

        self._recv = rtde_receive.RTDEReceiveInterface(host)
        self._lock = threading.Lock()
        self._bit = self.PORT_BIT0[str(port)] + int(pin)
        self.scene = scene
        self.closed = False
        self.edges: list[dict[str, Any]] = []
        self.level: Optional[bool] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tool-output", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 3.0
        while self.level is None and time.monotonic() < deadline:
            time.sleep(0.01)

    def tcp(self) -> Optional[np.ndarray]:
        with self._lock:
            pose = self._recv.getActualTCPPose()
        return _ur_pose_matrix(pose) if pose else None

    def joints_deg(self) -> list[float]:
        with self._lock:
            return [round(math.degrees(v), 2) for v in self._recv.getActualQ()]

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                bits = int(self._recv.getActualDigitalOutputBits())
            level = bool((bits >> self._bit) & 1)
            if self.level is None:
                self.level = level
            elif level != self.level:
                self.level = level
                self.closed = not self.closed
                tcp = self.tcp()
                self.edges.append({"t": time.time(), "level": level, "jaws": "closed" if self.closed else "open",
                                   "tcp_mm": None if tcp is None else [round(float(v), 1) for v in tcp[:3, 3]]})
                if tcp is not None:
                    self.scene.jaws_changed(self.closed, tcp)
            time.sleep(0.002)

    def changes_since(self, t0: float, t1: float = float("inf")) -> list[dict[str, Any]]:
        return [e for e in self.edges if t0 <= e["t"] <= t1]

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2.0)
        try:
            self._recv.disconnect()
        except Exception:  # noqa: BLE001 (a teardown never masks what was measured)
            pass


@dataclass
class Recorder:
    """Every command that reached the controller, every verb the arm was asked, every sample the guard judged, and
    every push the pick planned and drove, each with its wall-clock time."""

    commands: list[dict[str, Any]] = field(default_factory=list)
    verbs: list[dict[str, Any]] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    plans: list[dict[str, Any]] = field(default_factory=list)
    pushes: list[dict[str, Any]] = field(default_factory=list)
    floors: list[dict[str, Any]] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)
    on_command: list[Callable[[dict[str, Any]], None]] = field(default_factory=list)

    def verb_spy(self, verb: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def spy(*args: Any, **kwargs: Any) -> Any:
            row: dict[str, Any] = {"t": time.time(), "verb": verb,
                                   "linear": bool(kwargs.get("linear", False)),
                                   "vel": kwargs.get("vel"), "acc": kwargs.get("acc")}
            target = args[0] if args else None
            position = getattr(target, "position_mm", None)
            if position is not None:
                row["target_mm"] = [round(float(v), 2) for v in position]
            values = getattr(target, "values", None)
            if values is not None:
                row["target_deg"] = [round(math.degrees(float(v)), 2) for v in values]
            self.verbs.append(row)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                row["raised"] = f"{type(exc).__name__}: {exc}"[:300]
                row["t_end"] = time.time()
                raise
            row["t_end"] = time.time()
            status = getattr(result, "status", None)
            row["ok"] = bool(getattr(result, "ok", result))
            row["status"] = getattr(status, "value", None if status is None else str(status))
            row["message"] = str(getattr(result, "message", "") or "")[:400]
            return result

        return spy


def command_spy(original: Callable[..., Any], rows: list[dict[str, Any]],
                hooks: list[Callable[[dict[str, Any]], None]], verb: str) -> Callable[..., Any]:
    """What reaches the controller: each call of ``verb`` on the RTDE connection, recorded before it is sent (the hooks
    see it then), and what the controller answered."""

    def call(*args: Any, **kwargs: Any) -> Any:
        row: dict[str, Any] = {"t": time.time(), "verb": verb}
        if args:
            first = args[0]
            if verb == "moveL":
                row["tcp_mm"] = [round(float(v) * 1000.0, 2) for v in list(first)[:3]]
                row["pose"] = [float(v) for v in first]
            elif verb == "moveJ":
                row["joints_deg"] = [round(math.degrees(float(v)), 2) for v in list(first)[:6]]
            else:
                row["args"] = [repr(a)[:60] for a in args]
        if verb in ("moveL", "moveJ"):
            vel = kwargs.get("vel", args[1] if len(args) > 1 else None)
            acc = kwargs.get("acc", args[2] if len(args) > 2 else None)
            row["vel"] = None if vel is None else float(vel)
            row["acc"] = None if acc is None else float(acc)
        rows.append(row)
        for hook in list(hooks):
            hook(row)
        try:
            value = original(*args, **kwargs)
        except BaseException as exc:
            row["raised"] = f"{type(exc).__name__}: {exc}"[:200]
            row["t_end"] = time.time()
            raise
        row["t_end"] = time.time()
        row["result"] = repr(value)[:80]
        return value

    return call


# ---------------------------------------------------------------------------------------------------------------------
# The person at the cell: answers the toggle's question, unlocks a protective stop, decides after a stop
# ---------------------------------------------------------------------------------------------------------------------


def _dashboard(*commands: str) -> list[str]:
    s = socket.create_connection((URSIM_ADDRESS, 29999), timeout=10)
    s.settimeout(10)
    said = [s.recv(256).decode(errors="replace").strip()]
    for command in commands:
        s.sendall((command + "\n").encode())
        time.sleep(0.5)
        try:
            said.append(f"{command} -> " + s.recv(512).decode(errors="replace").strip())
        except socket.timeout:
            said.append(f"{command} -> (no answer)")
    s.close()
    return said


def _protective_stop_now() -> None:
    """A protective stop, as a secondary program raises it on the controller (URSim only)."""
    s = socket.create_connection((URSIM_ADDRESS, 30002), timeout=5)
    s.sendall(b"sec probe_stop():\n  protective_stop()\nend\n")
    s.close()


# ---------------------------------------------------------------------------------------------------------------------
# The URSim run
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Bed:
    """Everything a scenario reads: the cell, its service, the scene, the output, the recorder."""

    tree: Any
    cell: Any
    service: Any
    scene: ms.Scene
    output: ToolOutput
    rec: Recorder
    camera: Any
    backend: Any
    work: Path
    push_mm: float
    gap_mm: float
    look: Any
    at_xy: tuple[float, float] = CYLINDER_XY
    scenes: dict[str, tuple[str, float, tuple[float, float]]] = field(default_factory=dict)
    cancel: dict[str, Any] = field(default_factory=lambda: {"asked_at": None})
    #: The stand-ins of ``--push-stand-in``, each off until a scenario that is allowed them turns it on: the trigger
    #: forced (the plan's calculator stand-in) and the push planned on what cameras all round would see.
    force: dict[str, Any] = field(default_factory=lambda: {"trigger": False, "perception": False, "pushed": False,
                                                           "allowed": False})
    #: The scenario running, ``NAME`` or ``NAME#n``.
    current: str = ""

    def scene_of(self, scenario: str) -> tuple[str, float, tuple[float, float]]:
        """The neighbours, their gap and where the cylinder stands for ``scenario`` (``NAME`` or ``NAME#n``):
        ``--scene``, else the defaults."""
        return self.scenes.get(scenario, (SCENARIO_NEIGHBOUR[_base(scenario)], self.gap_mm, self.at_xy))

    @property
    def orchestrator(self) -> Any:
        return self.service.runtime.orchestrator


def _report_dict(report: Any) -> dict[str, Any]:
    pick = getattr(report, "pick_report", None)
    attempts = []
    for attempt in getattr(pick, "attempts", ()) or ():
        attempts.append({
            "index": getattr(attempt, "attempt_index", None), "action": getattr(attempt, "action", None),
            "reasons": [str(getattr(r, "value", r)) for r in (getattr(attempt, "reasons", ()) or ())],
            "push": getattr(attempt, "push", None), "blocker": getattr(attempt, "blocker", None),
            "motion_status": str(getattr(attempt, "motion_status", None)),
            "motion_message": str(getattr(attempt, "motion_message", "") or "")[:300],
            "tries": len(getattr(attempt, "tries", ()) or ()),
        })
    telemetry = getattr(report, "telemetry", {}) or {}
    return {
        "outcome": str(getattr(getattr(report, "outcome", None), "value", getattr(report, "outcome", None))),
        "pick_outcome": str(getattr(getattr(pick, "outcome", None), "value", None)),
        "failure_summary": _safe(lambda: report.failure_summary()),
        "needs_person": str(getattr(report, "needs_person", "") or ""),
        "push_stopped": str(telemetry.get("push_stopped") or ""),
        "attempts": attempts,
    }


def _safe(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception as exc:  # noqa: BLE001
        return f"({type(exc).__name__}: {exc})"


def _pushes(orchestrator: Any) -> list[dict[str, Any]]:
    return [p.to_dict() for p in getattr(orchestrator, "pushes", ()) or ()]


def _blockers(orchestrator: Any) -> list[Any]:
    out = []
    for b in getattr(orchestrator, "blockers", ()) or ():
        out.append(b.to_dict() if hasattr(b, "to_dict") else repr(b))
    return out


def _min_clearances(bed: Bed, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute, after the run, the least distance every sample the guard admitted keeps to every box the guard held
    then (the camera's at perceived_min_distance_mm, the declared at min_distance_mm): the guard's own meshes."""
    from src.robot.safety._fcl_self_collision import _yaw_matrix  # noqa: PLC0415
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: PLC0415

    out = []
    for row in rows:
        guard, model = row["guard"], row["model"]
        backend = guard._exact_mesh_backend(model)
        if backend is None:
            out.append({"t": row["t"], "least_mm": None, "why": "no exact backend"})
            continue
        transforms = ur_link_transforms_mm(model, np.asarray(row["joints"], dtype=np.float64))
        if transforms is None:
            out.append({"t": row["t"], "least_mm": None, "why": "no DH chain"})
            continue
        a = backend._a
        rb = _yaw_matrix(float(guard._config.kinematics_base_yaw_deg))
        for name in backend._names:
            t = transforms[backend._frame[name]]
            a.set_transform(backend._models[name], rb @ t[:3, :3], rb @ t[:3, 3])
        least, pair, kinds = math.inf, "", set()
        for fixtures, kind in ((row["perceived"], "seen"), (row["declared"], "declared")):
            for fx in fixtures:
                kinds.add(str(getattr(fx, "name", "")))
                centre = np.asarray(fx.center_mm, dtype=np.float64)
                turned = getattr(fx, "turned", None)
                if turned is None:
                    boxed = a.box_object(np.asarray(fx.half_extents_mm, dtype=np.float64), centre)
                else:
                    boxed = a.box_object(np.asarray(turned.half_extents_mm, dtype=np.float64), centre,
                                         float(turned.yaw_rad), getattr(turned, "rotation", None))
                for name in backend._names:
                    d = float(a.distance(backend._models[name], boxed))
                    if d < least:
                        least, pair = d, f"{name}|{kind}:{getattr(fx, 'name', '')}"
        out.append({"t": row["t"], "least_mm": round(least, 2), "pair": pair,
                    "support_held": any("support" in k for k in kinds), "bench_held": any("bench" in k for k in kinds),
                    "boxes": len(row["perceived"]) + len(row["declared"])})
    return out


def _within(rows: list[dict[str, Any]], t0: float, t1: float) -> list[dict[str, Any]]:
    return [r for r in rows if t0 <= r["t"] <= t1]


def _push_legs(bed: Bed, t0: float, t1: float) -> list[dict[str, Any]]:
    """The push's motions, P0 and the four contact legs, as the arm was asked them, with what the guard judged and
    what reached the controller in each."""
    window = [p for p in bed.rec.pushes if t0 <= p["t"] <= t1]
    if not window:
        return []
    push = window[0]
    verbs = [v for v in bed.rec.verbs if push["t"] <= v["t"] <= push["t_end"] and v["verb"] == "move"]
    names = ["P0", "down", "push", "back", "up"]
    legs = []
    for name, verb in zip(names, verbs):
        end = verb.get("t_end", verb["t"])
        sent = [c for c in bed.rec.commands if verb["t"] <= c["t"] <= end and c["verb"] in ("moveL", "moveJ")]
        judged = [s for s in bed.rec.samples if verb["t"] <= s["t"] <= end]
        legs.append({"leg": name, "linear": verb["linear"], "ok": verb.get("ok"), "status": verb.get("status"),
                     "message": verb.get("message", "")[:200], "vel": verb.get("vel"),
                     "target_mm": verb.get("target_mm"), "t": verb["t"], "t_end": end,
                     "commands": len(sent), "command_vel": [c.get("vel") for c in sent][:3],
                     "commanded_tcp_mm": [c.get("tcp_mm") for c in sent if c["verb"] == "moveL"],
                     "samples": len(judged), "rejected": sum(1 for s in judged if not s["accepted"]),
                     "_samples": judged})
    return legs


def _person_answer(bed_holder: dict[str, Any]) -> Callable[[str], str]:
    def answer(question: str) -> str:
        output = bed_holder.get("output")
        said = "closed" if output is not None and output.closed else "open"
        bed_holder["rec"].questions.append({"t": time.time(), "question": question[:200], "answer": said})
        return said

    return answer


def run_on_ursim(tree: Any, copied: Path, work: Path, crop: Path, *, scenarios: Sequence[str], push_mm: float,
                 gap_mm: float, at_xy: tuple[float, float], looks: str,
                 scenes: Optional[dict[str, tuple[str, float, tuple[float, float]]]] = None,
                 push_stand_in: bool = False) -> dict[str, Any]:
    """Build the cell from the copied tree with the stand-in camera and backend, connect it to URSim, and run each
    scenario as a whole pick of the cell's service."""
    from src.camera.orchestration import camera as camera_module  # noqa: PLC0415
    from src.models import perception_spec as spec_module  # noqa: PLC0415
    from src.robot.core import JointPositions  # noqa: PLC0415
    from src.robot.execution.cell import Cell  # noqa: PLC0415
    from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator  # noqa: PLC0415
    from src.robot.grasping.motion.grasp_motion import GraspMotion  # noqa: PLC0415
    from src.robot.grasping.recovery import push_motion, push_planner  # noqa: PLC0415
    from src.robot.grippers.jaw_io import JawIOGripper  # noqa: PLC0415
    from src.robot.safety.self_collision import SelfCollisionGuard  # noqa: PLC0415

    robot = tree.robot
    static = ms.StaticScene.from_crop(crop)
    scene = ms.Scene(static)
    rec = Recorder()
    holder: dict[str, Any] = {"rec": rec}
    calibration = _primary_calibration(tree)
    cam_to_tool = np.asarray(calibration.camera_to_tool().to_matrix(), dtype=np.float64)
    jaw = robot.gripper.jaw_io
    output = ToolOutput(URSIM_ADDRESS, str(getattr(jaw.io_port, "value", jaw.io_port)), int(jaw.close_output_pin),
                        scene)
    holder["output"] = output
    cameras: list[ms.StandInD415] = []

    def create_streamer(rig: Any) -> ms.StandInD415:
        camera = ms.StandInD415(rig, scene, camera_to_tool=cam_to_tool, tcp=output.tcp)
        cameras.append(camera)
        return camera

    backend_holder: dict[str, Any] = {}

    def spec_from_config(cls: Any, models: Any) -> ms.StandInSpec:
        backend = ms.StandInBackend(cameras[-1])
        backend_holder["backend"] = backend
        return ms.StandInSpec(backend)

    force: dict[str, Any] = {"trigger": False, "perception": False, "pushed": False, "allowed": push_stand_in}
    original_plan_push = push_planner.plan_push
    original_execute_push = push_motion.execute_push
    original_floor = BinPickingOrchestrator._finger_floor_mm
    original_evaluate = SelfCollisionGuard._evaluate_fcl

    def plan_push_spy(**kwargs: Any) -> Any:
        row: dict[str, Any] = {"t": time.time(), "kwargs": {k: v for k, v in kwargs.items()
                                                             if k not in ("target_points_mm", "neighbour_points_mm",
                                                                          "table_points_mm")}}
        table = kwargs.get("table_points_mm")
        row["table_points"] = None if table is None else np.asarray(table, dtype=np.float64)
        row["target_points"] = np.asarray(kwargs["target_points_mm"], dtype=np.float64)
        row["neighbour_points"] = np.asarray(kwargs["neighbour_points_mm"], dtype=np.float64)
        plan = original_plan_push(**kwargs)
        row["plan"] = plan
        if force["perception"] and not isinstance(plan, push_planner.PushPlan):
            # The stand-in of --push-stand-in: the planner asked again with what cameras all round would hand it, the
            # part's whole surface and the mat round it; every other input is the pick's own.
            target = next((part for part in scene.parts.values() if part.name == scene.target), None)
            completed = dict(kwargs)
            if target is not None:
                completed["target_points_mm"] = np.vstack([row["target_points"], target.surface_points()])
            table = row["table_points"] if row["table_points"] is not None else np.zeros((0, 3))
            completed["table_points_mm"] = np.vstack([table, scene.mat_points()])
            row["refused_on_the_frames"] = plan
            plan = original_plan_push(**completed)
            row["plan"] = plan
            row["stand_in"] = "the part's whole surface and the mat round it handed to plan_push"
        rec.plans.append(row)
        return plan

    def execute_push_spy(arm: Any, plan: Any, **kwargs: Any) -> Any:
        row: dict[str, Any] = {"t": time.time(), "plan": plan}
        rec.pushes.append(row)
        outcome = None
        try:
            outcome = original_execute_push(arm, plan, **kwargs)
            return outcome
        finally:
            row["t_end"] = time.time()
            row["outcome"] = outcome
            if outcome is not None and getattr(outcome, "plan", None) is not None and outcome.motion_started:
                scene.pushed(outcome.plan, tuple(outcome.legs_done), output.tcp())
                force["pushed"] = True

    def floor_spy(model: Any, points: Any, normal: Any, offset: Any, gate: Any) -> Any:
        value = original_floor(model, points, normal, offset, gate)
        rec.floors.append({"t": time.time(), "model": model, "points": np.asarray(points, dtype=np.float64),
                           "normal": np.asarray(normal, dtype=np.float64), "offset": float(offset), "floor": value})
        return value

    def evaluate_spy(guard: Any, ctx: Any) -> Any:
        decision = original_evaluate(guard, ctx)
        joints = getattr(getattr(ctx, "target_joints", None), "values", None)
        if joints is not None and ctx.arm is not None:
            rec.samples.append({"t": time.time(), "joints": [float(v) for v in joints], "guard": guard,
                                "model": guard.model_for(ctx.arm),
                                "accepted": decision is None or bool(decision.accepted),
                                "message": "" if decision is None else str(decision.message)[:200],
                                "perceived": tuple(guard._perceived_fixtures or ()),
                                "declared": tuple(guard._declared_fixtures or ())})
        return decision

    result: dict[str, Any] = {"scenarios": {}}
    look = [JointPositions.deg(*joints) for joints in _looks_deg(robot, looks)]
    motion = GraspMotion(standoff_mm=10.0, close_squeeze_mm=1.0, retreat_mm=100.0)
    with ExitStack() as stack:
        stack.callback(output.close)
        stack.enter_context(mock.patch.object(camera_module, "create_streamer", create_streamer))
        stack.enter_context(mock.patch.object(spec_module.PerceptionSpec, "from_config",
                                              classmethod(spec_from_config)))
        stack.enter_context(mock.patch.object(JawIOGripper, "_question", lambda self: _person_answer(holder)))
        stack.enter_context(mock.patch.object(push_planner, "plan_push", plan_push_spy))
        stack.enter_context(mock.patch.object(push_motion, "execute_push", execute_push_spy))
        stack.enter_context(mock.patch.object(BinPickingOrchestrator, "_finger_floor_mm", staticmethod(floor_spy)))
        stack.enter_context(mock.patch.object(SelfCollisionGuard, "_evaluate_fcl", evaluate_spy))
        cell = Cell.from_tree(tree, prompt=PROMPT, motion=motion)
        t_build = time.time()
        cell.build()
        result["build_s"] = round(time.time() - t_build, 1)
        if not isinstance(cell.arm, URRobotArm):
            raise RuntimeError(f"the tree built a {type(cell.arm).__name__}; this probe reads the UR driver's seams")
        with cell.connected():
            service = cell.service
            arm = cell.arm
            conn = arm.connection
            for verb in ("moveJ", "moveL", "stop", "set_digital_out"):
                setattr(conn, verb, command_spy(getattr(conn, verb), rec.commands, rec.on_command, verb))
            for verb in ("move", "move_to_joints", "move_to_joints_on_the_line", "move_linear", "move_joint"):
                if callable(getattr(arm, verb, None)):
                    setattr(arm, verb, rec.verb_spy(verb, getattr(arm, verb)))
            service.enable_record_logging(work / "records.jsonl", provenance={"probe": "probe_push_on_the_mat"})
            bed = Bed(tree=tree, cell=cell, service=service, scene=scene, output=output, rec=rec,
                      camera=cameras[0] if cameras else None, backend=backend_holder.get("backend"), work=work,
                      push_mm=push_mm, gap_mm=gap_mm, look=look, at_xy=at_xy, scenes=dict(scenes or {}))
            service.set_cancel_check(lambda: bed.cancel["asked_at"] is not None)
            bed.force = force
            _force_the_trigger(bed, force)
            for name in scenarios:
                print(f"\n=== scenario {name} ===", flush=True)
                bed.current = name
                try:
                    result["scenarios"][name] = SCENARIOS[_base(name)](bed)
                except Exception as exc:  # noqa: BLE001 (a scenario that raised is a failed item, said)
                    result["scenarios"][name] = {"held": False, "raised": f"{type(exc).__name__}: {exc}",
                                                 "traceback": traceback.format_exc()[-3000:]}
                print(json.dumps(_brief(result["scenarios"][name]), indent=1, default=str)[:4000], flush=True)
            result["camera"] = {"grabs": sum(c.grabs for c in cameras), "renders": sum(c.renders for c in cameras),
                                "render_s": round(sum(c.render_s for c in cameras), 1)}
            result["questions"] = rec.questions
            result["scene_events"] = scene.events
    return result


def _force_the_trigger(bed: Bed, force: dict[str, Any]) -> None:
    """The plan's calculator stand-in (Track P, P-1), used only by ``--push-stand-in``: while it is on and nothing has
    been pushed in the pick, the cell's calculator answers the part with no grasp, ``ALL_COLLIDED``, and hands on its own
    metadata (Track A's arrays, contract 2), so the pick's evidence finds the neighbour it saw. A blocker's own grasp
    (the segmentation the pick labels ``blocker``) is answered by the calculator itself."""
    import dataclasses  # noqa: PLC0415

    from src.robot.grasping.types.feedback import GraspFailureReason  # noqa: PLC0415

    calculator = bed.orchestrator.calculator
    original = calculator.compute_result

    def compute_result(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if not force["trigger"] or force["pushed"]:
            return result
        labels = [str(getattr(a, "label", "")) for a in (*args, *kwargs.values()) if hasattr(a, "mask")]
        if any(label == "blocker" for label in labels):
            return result
        telemetry = dict(getattr(result, "telemetry", {}) or {})
        telemetry["no_grasp_said"] = "stand-in: every grasp meets a neighbour (the probe forced the trigger)"
        telemetry["probe_forced_trigger"] = len(getattr(result, "candidates", ()) or ())
        return dataclasses.replace(result, candidates=(), reasons=(GraspFailureReason.ALL_COLLIDED,), top_score=0.0,
                                   telemetry=telemetry)

    calculator.compute_result = compute_result


def _brief(scenario: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in scenario.items() if k in ("held", "checks", "raised", "report")}


# ---------------------------------------------------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------------------------------------------------


def _base(scenario: str) -> str:
    """``NAME`` of ``NAME#n``: one scenario run again on a scene of its own."""
    return scenario.split("#", 1)[0]


def _start(bed: Bed, scenario: str) -> dict[str, Any]:
    neighbour, gap, at_xy = bed.scene_of(scenario)
    bed.cancel["asked_at"] = None
    stand_in = bool(bed.force.get("allowed")) and _base(scenario) in ("push", "cancel", "pstop")
    bed.force.update({"trigger": stand_in, "perception": stand_in, "pushed": False})
    bed.rec.on_command.clear()
    if bed.output.closed:
        bed.cell.gripper.set_closed(False)  # the last pick's part let go, before this scenario counts
        time.sleep(0.5)
    placed = place_scene(bed.scene, neighbour, gap, at_xy=at_xy)
    bed.service.start_campaign(push_mm=bed.push_mm)
    return placed


def _pick(bed: Bed) -> tuple[Any, float, float]:
    t0 = time.time()
    report = bed.service.pick(look=bed.look)
    return report, t0, time.time()


def _check(checks: dict[str, Any], name: str, held: bool, detail: Any = None) -> None:
    checks[name] = {"held": bool(held), "detail": detail}


def _toward_the_part(bed: Bed, t0: float, t1: float, at_xy: Sequence[float]) -> list[dict[str, Any]]:
    """The motions in the window that ended within 150 mm of the part (in plan) and under 250 mm over the mat."""
    centre = np.asarray(at_xy, dtype=np.float64)
    out = []
    for c in _within(bed.rec.commands, t0, t1):
        if c["verb"] == "moveL" and c.get("tcp_mm"):
            p = np.asarray(c["tcp_mm"], dtype=np.float64)
            if np.hypot(*(p[:2] - centre)) < 150.0 and p[2] < 250.0:
                out.append(c)
    for v in _within(bed.rec.verbs, t0, t1):
        if v.get("target_mm"):
            p = np.asarray(v["target_mm"], dtype=np.float64)
            if np.hypot(*(p[:2] - centre)) < 150.0 and p[2] < 250.0:
                out.append({"verb": v["verb"], "target_mm": v["target_mm"], "linear": v["linear"]})
    return out


def scenario_refusal(bed: Bed) -> dict[str, Any]:
    """P-2: the cylinder beside a block too wide to grip. No motion toward the part, the record says ALL_COLLIDED and
    no_free_direction, no change of the tool output."""
    placed = _start(bed, bed.current or "refusal")
    report, t0, t1 = _pick(bed)
    orch = bed.orchestrator
    rep = _report_dict(report)
    pushes = _pushes(orch)
    checks: dict[str, Any] = {}
    reasons = {r for a in rep["attempts"] for r in a["reasons"]}
    _check(checks, "all_collided", "all_collided" in reasons, sorted(reasons))
    codes = [p["code"] for p in pushes]
    _check(checks, "push_refused_before_motion", bool(codes) and not any(p.get("arm_moved") for p in pushes)
           and all(c.startswith(("no_", "part_", "refused_", "landing_")) for c in codes), codes)
    toward = _toward_the_part(bed, t0, t1, placed["target_foot_mm"][:2])
    _check(checks, "no_motion_toward_the_part", not toward and not any(p.get("motion_started") for p in pushes),
           toward[:5])
    changes = bed.output.changes_since(t0, t1)
    _check(checks, "no_do0_change", not changes, changes)
    record = _last_record(bed)
    text = json.dumps(record) if record else ""
    _check(checks, "record_says_all_collided_and_the_push_refusal",
           "all_collided" in text and bool(codes) and all(c in text for c in codes), _record_excerpt(record))
    return {"held": all(c["held"] for c in checks.values()), "checks": checks, "scene": placed, "report": rep,
            "pushes": pushes, "blockers": _blockers(orch), "seconds": round(t1 - t0, 1),
            "commands": len(_within(bed.rec.commands, t0, t1))}


def scenario_blocker(bed: Bed) -> dict[str, Any]:
    """R: the cylinder beside a 40 mm cube. The cube is gripped, set down on a free spot the camera saw, and the cylinder
    gripped; one change of the tool output per command."""
    placed = _start(bed, bed.current or "blocker")
    events_before = len(bed.scene.events)
    report, t0, t1 = _pick(bed)
    orch = bed.orchestrator
    rep = _report_dict(report)
    blockers = _blockers(orch)
    events = bed.scene.events[events_before:]
    checks: dict[str, Any] = {}
    set_aside = [b for b in blockers if isinstance(b, dict) and b.get("code") == "set_aside"]
    _check(checks, "blocker_set_aside", bool(set_aside), blockers)
    gripped = [e for e in events if e["event"] == "gripped"]
    downs = [e for e in events if e["event"] == "set_down"]
    order = [e.get("part") for e in gripped]
    blocker_names = {name for name in bed.scene.parts if name != TARGET}
    _check(checks, "a_neighbour_gripped_then_the_cylinder",
           len(order) >= 2 and order[0] in blocker_names and order[-1] == TARGET, order)
    cube_down = next((e for e in downs if e.get("part") in blocker_names), None)
    spot_ok = False
    if cube_down is not None:
        at = np.asarray(cube_down["at_mm"], dtype=np.float64)
        from_part = float(np.hypot(at[0] - placed["target_foot_mm"][0], at[1] - placed["target_foot_mm"][1]))
        mx0, mx1, my0, my1 = ms.MAT_XY
        on_mat = mx0 < at[0] < mx1 and my0 < at[1] < my1
        spot_ok = from_part >= 150.0 and on_mat and at[0] < BIN_X_MIN_MM
        cube_down = {**cube_down, "from_the_part_mm": round(from_part, 1), "on_the_mat": on_mat}
    _check(checks, "set_down_on_a_free_spot_150mm_from_the_part", spot_ok, cube_down)
    changes = bed.output.changes_since(t0, t1)
    jaws = [c["jaws"] for c in changes]
    _check(checks, "do0_one_change_per_command", jaws == ["closed", "open", "closed"], changes)
    _check(checks, "target_gripped_at_the_end", rep["outcome"] == "succeeded" and bool(order) and order[-1] == TARGET,
           {"outcome": rep["outcome"], "pick": rep["pick_outcome"]})
    rows = [a for a in rep["attempts"] if a["action"] == "clear_blocker"]
    _check(checks, "attempt_row_clear_blocker", bool(rows), rows)
    return {"held": all(c["held"] for c in checks.values()), "checks": checks, "scene": placed, "report": rep,
            "blockers": blockers, "pushes": _pushes(orch), "scene_events": events, "seconds": round(t1 - t0, 1),
            "commands": len(_within(bed.rec.commands, t0, t1))}


def _push_checks(bed: Bed, t0: float, t1: float, checks: dict[str, Any]) -> dict[str, Any]:
    """Items 1-4 and 8 on the push in the window."""
    orch = bed.orchestrator
    plans = [p for p in bed.rec.plans if t0 <= p["t"] <= t1]
    from src.robot.grasping.recovery.push_planner import PushPlan  # noqa: PLC0415

    planned = [p for p in plans if isinstance(p["plan"], PushPlan)]
    detail: dict[str, Any] = {"plans_asked": len(plans), "planned": len(planned)}
    if not planned:
        _check(checks, "item1_plan_on_the_mat", False, {"refusals": [getattr(p["plan"], "code", None) for p in plans]})
        return detail
    row = planned[0]
    plan = row["plan"]
    normal = np.asarray(row["kwargs"]["support_normal"], dtype=np.float64)
    tilt = math.degrees(math.acos(abs(float(normal[2])) / float(np.linalg.norm(normal))))
    table = row["table_points"]
    landing = np.asarray(plan.landing_box_xy_mm, dtype=np.float64)
    mx0, mx1, my0, my1 = ms.MAT_XY
    inside = bool(landing[0][0] >= mx0 and landing[1][0] <= mx1 and landing[0][1] >= my0 and landing[1][1] <= my1)
    clear_of_bin = bool(landing[1][0] < BIN_X_MIN_MM)
    table_on_mat = None
    if table is not None and len(table):
        mat = bed.scene.static
        heights = table[:, 2] - (mat.mat_plane[0] + mat.mat_plane[1] * table[:, 0] + mat.mat_plane[2] * table[:, 1])
        table_on_mat = {"points": int(table.shape[0]), "off_the_mat_p95_mm": round(float(np.percentile(
            np.abs(heights), 95)), 2), "x_max": round(float(table[:, 0].max()), 1)}
    detail["plan"] = plan.to_dict()
    detail["support_tilt_deg"] = round(tilt, 2)
    detail["support_offset_mm"] = round(float(row["kwargs"]["support_offset_mm"]), 2)
    detail["finger_floor_mm"] = row["kwargs"].get("finger_floor_mm")
    detail["landing_box_xy_mm"] = landing.round(1).tolist()
    detail["table"] = table_on_mat
    _check(checks, "item1_plan_on_the_mat", 1.0 <= tilt <= 1.2 and inside and clear_of_bin,
           {"tilt_deg": round(tilt, 2), "landing_inside_the_mat": inside, "clear_of_the_bin": clear_of_bin,
            "table": table_on_mat})
    legs = _push_legs(bed, t0, t1)
    clearances: list[dict[str, Any]] = []
    for leg in legs:
        judged = [s for s in leg.pop("_samples") if s["accepted"]]
        mins = _min_clearances(bed, judged)
        leg["least_mm"] = min((m["least_mm"] for m in mins if m["least_mm"] is not None), default=None)
        leg["support_held"] = all(m.get("support_held") for m in mins) if mins else False
        leg["bench_held"] = all(m.get("bench_held") for m in mins) if mins else False
        leg["least_pair"] = min(mins, key=lambda m: m["least_mm"] or math.inf)["pair"] if mins else ""
        clearances.extend(mins)
    detail["legs"] = legs
    p0 = legs[0] if legs else {}
    contact = legs[1:5]
    item2 = (bool(p0) and bool(p0.get("ok")) and p0.get("samples", 0) > 0 and len(contact) == 4
             and all(leg.get("ok") and leg["samples"] >= 2 and leg["rejected"] == 0 and leg["support_held"]
                     and leg["bench_held"] and (leg["least_mm"] or 0.0) >= 5.0 for leg in contact))
    push_leg = next((leg for leg in contact if leg["leg"] == "push"), None)
    speed_ok = push_leg is not None and all(abs((v or 0.0) - PUSH_LEG_SPEED_M_S) < 1e-6 for v in push_leg["command_vel"])
    _check(checks, "item2_every_leg_judged_at_5mm", item2 and speed_ok,
           [{k: leg.get(k) for k in ("leg", "ok", "samples", "rejected", "least_mm", "least_pair", "support_held",
                                     "bench_held", "command_vel")} for leg in legs])
    floors = [f for f in bed.rec.floors if t0 <= f["t"] <= t1]
    item3: dict[str, Any] = {}
    if floors and contact:
        floor = floors[-1]
        model, n, off = floor["model"], floor["normal"], floor["offset"]
        from src.robot.grasping.recovery.push_gate import PushCell  # noqa: PLC0415

        cell_inputs = PushCell.from_robot_config(bed.tree.robot)
        hand = cell_inputs.hand if isinstance(cell_inputs, PushCell) else None
        depth = float(getattr(hand, "fingertip_depth_mm", 0.0) or 0.0)
        approach = np.asarray(plan.approach, dtype=np.float64)
        approach = approach / float(np.linalg.norm(approach))
        tips = []
        for leg in contact[:3]:
            for tcp in leg["commanded_tcp_mm"]:
                tip = np.asarray(tcp, dtype=np.float64) + depth * approach
                tips.append(tip)
        tops: list[float] = []
        if tips:
            path = np.asarray(tips)[:, :2]
            for solid in getattr(model, "solids", ()) or ():
                covered = solid.covers_xy(path)
                if covered.any():
                    tops.extend((np.column_stack([path[covered], solid.top_at(path[covered])]) @ n - off).tolist())
        lowest = min(((np.asarray(t) @ n) - off for t in tips), default=None)
        highest_top = max(tops, default=None)
        item3 = {"fingertip_depth_mm": depth, "lowest_fingertip_over_the_plane_mm": None if lowest is None else round(
            float(lowest), 2), "highest_solid_top_under_the_path_mm": None if highest_top is None else round(
            float(highest_top), 2), "finger_floor_mm": None if floor["floor"] is None else round(float(floor["floor"]), 2),
            "plan_finger_height_mm": round(float(plan.finger_height_mm), 2)}
        ok3 = lowest is not None and highest_top is not None and lowest >= highest_top + 5.0 - 0.05
        _check(checks, "item3_finger_over_the_solid_plus_5", ok3, item3)
    else:
        _check(checks, "item3_finger_over_the_solid_plus_5", False, {"floors": len(floors), "legs": len(contact)})
    detail["item3"] = item3
    pushes = [p for p in bed.rec.pushes if t0 <= p["t"] <= t1]
    if pushes:
        p = pushes[0]
        before = bed.output.changes_since(t0, p["t_end"])
        per_leg = [len(bed.output.changes_since(t0, leg["t"])) for leg in contact]
        _check(checks, "item4_no_do0_change_through_the_push", not before and not any(per_leg),
               {"changes_until_push_end": before, "count_before_each_leg": per_leg})
    record = _last_record(bed)
    text = json.dumps(record) if record else ""
    log_text = _robot_log(bed)
    wanted = {"trigger": "all_collided", "direction": "direction", "distance": "distance_mm",
              "finger_height": "finger_height", "landing_box": "landing_box", "leg_verdicts": "legs_done"}
    in_record = {k: (w in text) for k, w in wanted.items()}
    in_log = {"pushing the part": "pushing the part" in log_text, "the push plans on": "the push plans on" in log_text}
    _check(checks, "item8_record_and_log_carry_the_push", all(in_record.values()) and all(in_log.values()),
           {"record": in_record, "robot_log": in_log, "record_excerpt": _record_excerpt(record)})
    detail["pushes_recorded"] = _pushes(orch)
    return detail


def _stand_in_said(bed: Bed, t0: float, t1: float) -> dict[str, Any]:
    """Whether the scenario ran with ``--push-stand-in``, and what the planner said on the pick's own inputs first."""
    if not bed.force.get("trigger"):
        return {"used": False}
    rows = [p for p in bed.rec.plans if t0 <= p["t"] <= t1]
    first = [p.get("refused_on_the_frames") for p in rows if p.get("refused_on_the_frames") is not None]
    return {"used": True, "trigger_forced": True, "plan_push_on_the_frames": [
        getattr(f, "code", None) for f in first],
        "plan_push_on_the_frames_said": [getattr(f, "sentence", "")[:300] for f in first]}


def _last_record(bed: Bed) -> Optional[dict[str, Any]]:
    path = bed.work / "records.jsonl"
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def _record_excerpt(record: Optional[dict[str, Any]]) -> Any:
    if not record:
        return None
    extra = record.get("extra") or {}
    keep = {k: extra.get(k) for k in ("attempts", "pushes", "push", "recovery_actions", "failure") if k in extra}
    return {"outcome": record.get("outcome"), "failure_reason": record.get("failure_reason"), "extra": keep,
            "keys": sorted(record.keys())}


def _robot_log(bed: Bed) -> str:
    path = bed.work / "logs" / "robot" / "robot.log"
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def scenario_push(bed: Bed) -> dict[str, Any]:
    """P-1, items 1-4, 7 and 8: the cylinder beside a 25 mm cube nothing can grip. The push, a judged move back to the
    look, a fresh look, and the pick goes on."""
    placed = _start(bed, bed.current or "push")
    events_before = len(bed.scene.events)
    grabs_before = len(bed.backend.calls) if bed.backend is not None else 0
    report, t0, t1 = _pick(bed)
    orch = bed.orchestrator
    rep = _report_dict(report)
    checks: dict[str, Any] = {}
    blockers = _blockers(orch)
    _check(checks, "blocker_first_none_graspable",
           bool(blockers) and all(isinstance(b, dict) and b.get("code") in ("no_graspable_blocker",
                                                                             "no_blocker_frees_a_grasp")
                                  for b in blockers), blockers)
    detail = _push_checks(bed, t0, t1, checks)
    pushes = _pushes(orch)
    moved = [p for p in pushes if p.get("motion_started")]
    pushed = [p for p in pushes if p.get("code") == "pushed"]
    events = bed.scene.events[events_before:]
    after = [p for p in bed.rec.pushes if t0 <= p["t"] <= t1]
    back = []
    looked = 0
    if after:
        end = after[0]["t_end"]
        back = [v for v in bed.rec.verbs if v["t"] >= end and v["verb"] in ("move_to_joints", "move_to_joints_on_the_line",
                                                                              "move")][:1]
        if bed.backend is not None:
            looked = sum(1 for c in bed.backend.calls[grabs_before:] if c["t"] >= end)
    gripped = [e.get("part") for e in events if e["event"] == "gripped"]
    _check(checks, "item7_back_to_the_look_fresh_look_pick_goes_on",
           bool(pushed) and bool(back) and bool(back[0].get("ok")) and looked > 0 and gripped[-1:] == [TARGET]
           and rep["outcome"] == "succeeded",
           {"pushed": len(pushed), "move_back": back[:1], "fresh_looks": looked, "gripped": gripped,
            "outcome": rep["outcome"]})
    _check(checks, "item7_one_push_per_part_two_per_pick", len(moved) <= 1, {"pushes_that_moved": len(moved)})
    changes = bed.output.changes_since(t0, t1)
    _check(checks, "do0_only_the_final_grasp", [c["jaws"] for c in changes] == ["closed"], changes)
    return {"held": all(c["held"] for c in checks.values()), "checks": checks, "scene": placed, "report": rep,
            "push": detail, "blockers": blockers, "scene_events": events, "seconds": round(t1 - t0, 1),
            "commands": len(_within(bed.rec.commands, t0, t1)), "stand_in": _stand_in_said(bed, t0, t1)}


def _stop_during_the_push_leg(bed: Bed, how: str) -> dict[str, Any]:
    """Items 5 and 6: a stop while the push leg runs, then the next pick of the same run, then a new run."""
    placed = _start(bed, bed.current or how)
    fired: dict[str, Any] = {}

    def hook(row: dict[str, Any]) -> None:
        if fired or row["verb"] != "moveL" or abs((row.get("vel") or 0.0) - PUSH_LEG_SPEED_M_S) > 1e-6:
            return
        fired["leg_sent_at"] = row["t"]

        def later() -> None:
            time.sleep(STOP_INTO_THE_LEG_S)
            fired["at"] = time.time()
            if how == "cancel":
                bed.cancel["asked_at"] = fired["at"]
            else:
                _protective_stop_now()

        threading.Thread(target=later, daemon=True).start()

    bed.rec.on_command.append(hook)
    report, t0, t1 = _pick(bed)
    bed.rec.on_command.clear()
    orch = bed.orchestrator
    rep = _report_dict(report)
    checks: dict[str, Any] = {}
    _check(checks, "stop_fired_during_the_push_leg", "at" in fired, fired)
    at = fired.get("at", t1)
    leg_row = next((c for c in bed.rec.commands if c["t"] == fired.get("leg_sent_at")), None)
    later = [c for c in bed.rec.commands if c["t"] > at and c["verb"] in ("moveL", "moveJ") and c is not leg_row]
    _check(checks, "no_later_command", not later,
           [{k: c.get(k) for k in ("verb", "tcp_mm", "joints_deg", "vel")} for c in later[:8]] + (
               [f"... {len(later)} in all"] if len(later) > 8 else []))
    tcp_now = bed.output.tcp()
    where = None if tcp_now is None else [round(float(v), 1) for v in tcp_now[:3, 3]]
    latched = str(bed.service.stopped_where_the_arm_stands or "")
    _check(checks, "needs_person_latched", bool(latched), latched[:400])
    person: dict[str, Any] = {}
    if how == "pstop":
        time.sleep(5.5)  # the controller lets a protective stop go five seconds after it, at the earliest
        person["unlock"] = _dashboard("close safety popup", "unlock protective stop", "safetymode")
        time.sleep(1.0)
    commands_before = len(bed.rec.commands)
    changes_before = len(bed.output.edges)
    grabs_before = bed.camera.grabs if bed.camera is not None else 0
    bed.cancel["asked_at"] = None
    again = bed.service.pick(look=bed.look)
    again_rep = _report_dict(again)
    refused = again_rep["outcome"] == "unsafe_recovery_refused"
    quiet = (len(bed.rec.commands) == commands_before and len(bed.output.edges) == changes_before
             and (bed.camera is None or bed.camera.grabs == grabs_before))
    _check(checks, "next_pick_refused_until_a_new_run", refused and quiet,
           {"outcome": again_rep["outcome"], "nothing_commanded_or_perceived": quiet})
    bed.service.start_campaign(push_mm=bed.push_mm)
    person["new_run_clears"] = not bool(bed.service.stopped_where_the_arm_stands)
    pushes = _pushes(orch)
    return {"held": all(c["held"] for c in checks.values()), "checks": checks, "scene": placed, "report": rep,
            "pushes": pushes, "arm_stands_mm": where, "person": person, "seconds": round(t1 - t0, 1),
            "commands_after_the_stop": len(later), "stand_in": _stand_in_said(bed, t0, t1),
            "commands": [{k: c.get(k) for k in ("t", "verb", "tcp_mm", "joints_deg", "vel", "raised", "result")}
                         for c in bed.rec.commands if t0 <= c["t"] <= t1][-40:]}


def scenario_cancel(bed: Bed) -> dict[str, Any]:
    return _stop_during_the_push_leg(bed, "cancel")


def scenario_pstop(bed: Bed) -> dict[str, Any]:
    return _stop_during_the_push_leg(bed, "pstop")


SCENARIOS: dict[str, Callable[[Bed], dict[str, Any]]] = {
    "refusal": scenario_refusal, "blocker": scenario_blocker, "push": scenario_push, "cancel": scenario_cancel,
    "pstop": scenario_pstop,
}


# ---------------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------------


def _controller_up() -> Optional[str]:
    """Why URSim cannot be driven now, or ``None``: its dashboard answers, the arm runs, the safety mode is normal."""
    try:
        said = _dashboard("robotmode", "safetymode")
    except OSError as exc:
        return f"URSim's dashboard at {URSIM_ADDRESS}:29999 does not answer ({exc}): start it (scripts/ursim/ursim.sh)"
    text = " ".join(said)
    if "RUNNING" not in text or "NORMAL" not in text:
        return f"URSim is not ready to move: {said[1:]}; power it on and release the brakes first"
    return None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/ursim/probe_push_on_the_mat.py",
                                 description="The push's simulation gate on URSim with a cell's own tree.")
    ap.add_argument("--tree", required=True, help="the cell's config folder; it is copied, never written")
    ap.add_argument("--work", required=True, help="a new folder for the copy, the logs, the record and the result")
    ap.add_argument("--profile", default=URSIM_LAYER, help=f"the chain to load; it must name {URSIM_LAYER!r}")
    ap.add_argument("--calibration", default=None, help="the primary rig's artifact on this box (a camera layer)")
    ap.add_argument("--crop", default=str(DEFAULT_CROP), help="Track K's recorded look the scene is rendered into")
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--push-mm", type=float, default=50.0, help="the push distance (example 12 passes 50)")
    ap.add_argument("--gaps-mm", default="8,5,10,12", help="the cube's gaps step 0 tries, in order")
    ap.add_argument("--at", default=f"{CYLINDER_XY[0]:g},{CYLINDER_XY[1]:g}",
                    help="where the cylinder stands, BASE x,y mm")
    ap.add_argument("--looks", choices=("all", "first"), default="all",
                    help="the tree's looks each pick visits (example 12: all three), or LOOK[0] alone")
    ap.add_argument("--scene", action="append", default=[],
                    help="a scenario's own scene, NAME=NEIGHBOURS@GAP@X,Y, NEIGHBOURS name[:side[:gap]],... (for "
                         "example blocker=cube40@8@-150,-760 or blocker=cube40:-x,block60:+y:14@8@-150,-760)")
    ap.add_argument("--push-stand-in", action="store_true",
                    help="push, cancel and pstop force the trigger (the plan's calculator stand-in) and hand plan_push "
                         "the part's whole surface and the mat round it; said in the result")
    ap.add_argument("--offline", action="store_true", help="step 0 only: nothing connects")
    args = ap.parse_args(argv)

    chain = str(args.profile)
    if URSIM_LAYER not in _layers(chain):
        print(f"REFUSED: the profile chain {chain!r} does not name the {URSIM_LAYER!r} layer this probe writes, so "
              f"the controller it would reach is whatever the tree names. Nothing was copied or connected.")
        return _REFUSED
    work = Path(args.work)
    if work.exists() and any(work.iterdir()):
        print(f"REFUSED: {work} exists and is not empty; a run writes its copy, logs and result into a new folder. "
              "Nothing was copied or connected.")
        return _REFUSED
    scenarios = [s for s in _layers(args.scenarios)]
    scenes: dict[str, tuple[str, float, tuple[float, float]]] = {}
    for text in args.scene:
        try:
            name, rest = text.split("=", 1)
            neighbour, gap, where = rest.split("@")
            x, y = (float(v) for v in where.split(","))
            neighbours_of(neighbour)
            if _base(name) not in SCENARIOS:
                raise ValueError(text)
            scenes[name] = (neighbour, float(gap), (x, y))
        except ValueError:
            print(f"REFUSED: --scene {text!r} is not NAME=NEIGHBOURS@GAP@X,Y with a known scenario, neighbours and sides "
                  f"({sorted(SCENARIOS)}; {sorted(NEIGHBOURS)}; {list(SIDES)})")
            return _REFUSED
    unknown = [s for s in scenarios if _base(s) not in SCENARIOS]
    if unknown:
        print(f"REFUSED: unknown scenario(s) {unknown}; known: {sorted(SCENARIOS)}")
        return _REFUSED
    work.mkdir(parents=True, exist_ok=True)
    copied = copy_the_tree(args.tree, work, calibration=args.calibration)
    said = refusal_of(copied, chain)
    if said is not None:
        print(f"REFUSED: {said}. Nothing connected.")
        return _REFUSED
    os.environ["WILLY_LOG_DIR"] = str((work / "logs").resolve())
    tree = load_tree(chain, root=copied)
    if not tree.ok:
        print(f"REFUSED: the copied tree does not load under {chain!r}: {tree.error}. Nothing connected.")
        return _REFUSED
    result: dict[str, Any] = {"tree": str(args.tree), "copy": str(copied), "chain": chain, "address": URSIM_ADDRESS,
                              "push_mm": args.push_mm, "looks": args.looks, "scenes": args.scene,
                              "push_stand_in": bool(args.push_stand_in),
                              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    gaps = [float(g) for g in _layers(args.gaps_mm)]
    print("step 0: the push scene offline, S's model and plan_push as the pick asks it", flush=True)
    at_xy = tuple(float(v) for v in _layers(args.at))
    step0 = offline_step(tree, Path(args.crop), push_mm=float(args.push_mm), gaps=gaps, at_xy=at_xy,
                         looks=args.looks)
    result["step0"] = step0
    for row in step0.get("gaps", []):
        print(f"  gap {row['gap_mm']:g} mm: planned={row.get('planned')} tilt={row.get('surface_tilt_deg')} "
              f"floor={row.get('finger_floor_mm')} {row.get('refusal', {}).get('code', '') if row.get('refusal') else ''}",
              flush=True)
    out = work / "p_result.json"
    if not step0.get("planned"):
        print("step 0: no gap planned a push from the declared looks alone; the picks on URSim add the one view they "
              "generate, so the gate runs on and says what the whole pick planned")
    if args.offline:
        out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        return _OK if step0.get("planned") else _FAILED
    not_up = _controller_up()
    if not_up is not None:
        print(f"REFUSED: {not_up}. Nothing connected beyond the dashboard.")
        out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        return _REFUSED
    try:
        result["ursim"] = run_on_ursim(tree, copied, work, Path(args.crop), scenarios=scenarios,
                                       push_mm=float(args.push_mm), gap_mm=float(step0.get("gap_mm", gaps[0])),
                                       at_xy=(at_xy[0], at_xy[1]), looks=args.looks, scenes=scenes,
                                       push_stand_in=bool(args.push_stand_in))
    except Exception as exc:  # noqa: BLE001 (a bed that did not come up is a failed gate, said with its traceback)
        result["ursim"] = {"raised": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
    runs = (result.get("ursim") or {}).get("scenarios") or {}
    failed = {name: [k for k, c in run.get("checks", {}).items() if not c["held"]] or [run.get("raised", "")]
              for name, run in runs.items() if not run.get("held")}
    if "raised" in (result.get("ursim") or {}):
        failed["bed"] = [result["ursim"]["raised"]]
    result["failed"] = failed
    result["held"] = not failed and len(runs) == len(scenarios)
    out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\nresult: {out}")
    print("EVERY ITEM HELD" if result["held"] else f"NOT HELD: {json.dumps(failed, default=str)[:2000]}")
    return _OK if result["held"] else _FAILED


if __name__ == "__main__":
    raise SystemExit(main())
