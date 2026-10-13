"""The replay bank: the cell's recorded picks judged again by the code of today, offline (the owner, 2026-10-12).

A pick the cell recorded (``robot.grasping.record_views``) holds every look it took: the colour, the measured and the
surface depth, the camera's matrix, CAMERA and TOOL to BASE, the joints, the segmentations the detector gave (from
2026-10-12 on) and the target's cloud fused over the looks. The bank builds the owner's cell from a config tree, the UR
driver on the bench's instant controller (``scripts/bench/instant_ur.py``) with its preflight and exact guard, the
camera world and the calculator, and for every recorded pick asks what the cell would do with those frames now:

* the camera world of all its looks, the target held out as the cell holds it;
* the target's grasps on its last look, ranked by the pick's own orchestrator (its support, its neighbours, the parts
  beside), as many as the cell tries (``MAX_TRIES``);
* each grasp's line down from the policy's standoff and its lift, every sample solved nearest the one before and judged
  by the cell's own path gate (``SafetyPreflight.gate_joint_path``) in the world of that grasp, the space between its
  jaws left out as the arm leaves it.

What it says per pick: how many grasps the calculator offered and why none where none, how many the guard admits, the
first admitted and what refused the others. A bank run is the measure every lever and every port is held to: a change
that admits fewer grasps on the bank says so before a cell does. Nothing moves and nothing is sent: the controller is a
stand-in, cuRobo is not asked (where only a camera's box refuses, the exact guard decides on the cell too), and a look
the recording lacks is never made up.

A pick recorded before 2026-10-12 carries no segmentation: its target's mask is its fused cloud projected into the look
(``RecordedPick.target_mask``), and it has no neighbour the detector named.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts" / "bench") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts" / "bench"))

#: Where the recordings and the cell's config tree lie: ``<bank>/<day>/views/*.npz`` and ``<bank>/robominds_config``.
BANK = Path(os.environ.get("WILLY_REPLAY_BANK", "D:/dev/willy_replay_bank"))
#: How many grasps of a look the cell tries at most (``pick_loop.GRASPS_TRIED_PER_ATTEMPT``: the best and up to 23 more).
MAX_TRIES = 24
#: The stamp the recorded frames are given, seconds, and how much later each world is asked for.
_STAMP_S = 1000.0
_ASKED_AFTER_S = 0.05
#: How far a projected cloud point may stand from the look's own depth there and still be the target's, millimetres.
_ON_THE_TARGET_MM = 12.0


@dataclass(frozen=True)
class Look:
    """One recorded look: what the wrist camera saw at one pose, and what the detector named in it."""

    name: str
    joints_rad: tuple[float, ...]
    rgb: "np.ndarray | None"
    depth_mm: np.ndarray
    surface_depth_mm: np.ndarray
    intrinsics: np.ndarray
    camera_to_base: np.ndarray
    tool_to_base: np.ndarray
    masks: tuple[np.ndarray, ...] = ()
    labels: tuple[str, ...] = ()
    target: "int | None" = None

    @property
    def camera_to_tool(self) -> np.ndarray:
        return np.linalg.inv(self.tool_to_base) @ self.camera_to_base


@dataclass(frozen=True)
class RecordedPick:
    """A pick the cell recorded: its looks in the order taken and its target's cloud fused over them, BASE mm."""

    name: str
    day: str
    looks: tuple[Look, ...]
    target_cloud_mm: np.ndarray

    @property
    def target_label(self) -> str:
        """The detector's word for the target on the last look, ``"target"`` where it named none there."""
        last = self.looks[-1]
        return last.labels[last.target] if last.target is not None and last.target < len(last.labels) else "target"

    def target_mask(self, look: Look) -> np.ndarray:
        """The target's mask in ``look``: the detector's where it was recorded, else the fused cloud projected into the
        look, kept where the look's own depth stands within :data:`_ON_THE_TARGET_MM` of it and closed over the gaps."""
        import cv2  # noqa: PLC0415

        if look.target is not None and 0 <= look.target < len(look.masks):
            return look.masks[look.target]
        shape = look.depth_mm.shape[:2]
        points = np.c_[self.target_cloud_mm, np.ones(len(self.target_cloud_mm))] @ np.linalg.inv(look.camera_to_base).T
        z = points[:, 2]
        k = look.intrinsics
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(k[0, 0] * points[:, 0] / z + k[0, 2]).astype(np.int64)
            v = np.round(k[1, 1] * points[:, 1] / z + k[1, 2]).astype(np.int64)
        inside = (z > 0) & (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
        u, v, z = u[inside], v[inside], z[inside]
        measured = look.surface_depth_mm[v, u]
        seen = np.abs(measured - z) <= _ON_THE_TARGET_MM
        mask = np.zeros(shape, dtype=np.uint8)
        mask[v[seen], u[seen]] = 1
        return np.asarray(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))).astype(bool)


def _unpacked(packed: np.ndarray, shape: "tuple[int, int]") -> tuple[np.ndarray, ...]:
    count = shape[0] * shape[1]
    return tuple(np.unpackbits(row)[:count].reshape(shape).astype(bool) for row in np.asarray(packed))


def load_pick(path: "str | Path") -> RecordedPick:
    """One recorded pick (``record_views``' npz, either format)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        names = [str(v) for v in data["views"]] if "views" in data.files else []
        labels = [str(v) for v in data["labels"]] if "labels" in data.files else []
        looks = []
        index = 0
        while f"depth_{index}" in data.files:
            depth = np.asarray(data[f"depth_{index}"], dtype=np.float64)
            surface = np.asarray(data[f"surface_depth_{index}"], dtype=np.float64)
            for array in (depth, surface):
                array[array >= 65000.0] = 0.0
            joints = tuple(np.radians([float(v) for v in re.findall(r"-?\d+\.?\d*", labels[index])[:6]])) \
                if index < len(labels) else ()
            masks: tuple[np.ndarray, ...] = ()
            seg_labels: tuple[str, ...] = ()
            target = None
            if f"seg_masks_{index}" in data.files:
                masks = _unpacked(data[f"seg_masks_{index}"], depth.shape[:2])
                seg_labels = tuple(str(v) for v in data[f"seg_labels_{index}"])
                said = int(data[f"seg_target_{index}"])
                target = said if 0 <= said < len(masks) else None
            looks.append(Look(
                name=names[index] if index < len(names) else f"look{index}", joints_rad=tuple(float(v) for v in joints),
                rgb=np.asarray(data[f"rgb_{index}"]) if f"rgb_{index}" in data.files else None, depth_mm=depth,
                surface_depth_mm=surface, intrinsics=np.asarray(data[f"intrinsics_{index}"], dtype=np.float64),
                camera_to_base=np.asarray(data[f"camera_to_base_{index}"], dtype=np.float64),
                tool_to_base=np.asarray(data[f"tool_to_base_{index}"], dtype=np.float64),
                masks=masks, labels=seg_labels, target=target))
            index += 1
        cloud = np.asarray(data["target_cloud_base_mm"], dtype=np.float64).reshape(-1, 3) \
            if "target_cloud_base_mm" in data.files else np.zeros((0, 3))
    return RecordedPick(name=path.stem, day=path.parent.parent.name, looks=tuple(looks), target_cloud_mm=cloud)


def recorded_picks(bank: Path = BANK, days: "tuple[str, ...]" = ()) -> list[Path]:
    """Every recording of the bank, the days named or all, oldest first."""
    folders = [bank / day / "views" for day in days] if days else sorted(bank.glob("*/views"))
    return sorted(path for folder in folders for path in folder.glob("*.npz"))


@dataclass
class _Seg:
    mask: np.ndarray
    label: str
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class _RecordedResolver:
    """CAMERA to BASE of each recorded frame, as the look recorded it."""

    def __init__(self) -> None:
        self.by_frame: dict[int, np.ndarray] = {}

    def camera_to_base_for_frame(self, frame: Any, *, arm: Any = None) -> Any:
        from src.geometry import Frame, Transform  # noqa: PLC0415

        return Transform.from_matrix(self.by_frame[id(frame)], from_frame=Frame.CAMERA, to_frame=Frame.BASE)


class _NoCamera:
    """The perception source the service is built with: the bank hands it frames itself, so nothing is acquired."""

    def acquire(self, **_: Any) -> Any:
        raise RuntimeError("the replay bank ranks recorded frames; it acquires none")


#: What a bank run may switch off to set a lever beside the code without it (``ReplayCell(off=...)``): the ground
#: seeded round a held target (``support_surfaces._ground_round``), the parts beside the named ones read off the
#: depth (``robot.grasping.name_neighbours``), and the fine world the guard holds about a motion's goal
#: (``safety.planning.fine_world``) with the 2 mm grid SFE keeps the hand off the neighbours' points by.
LEVERS = ("ground", "parts_beside", "fine_world")


class ReplayCell:
    """The owner's cell from ``config_dir``, offline: its UR driver on the instant controller, preflight and exact
    guard, camera world, calculator, orchestrator and policy; ``off`` the :data:`LEVERS` it runs without."""

    def __init__(self, config_dir: "str | Path" = BANK / "robominds_config" / "config",
                 off: "tuple[str, ...]" = ()) -> None:
        import instant_ur  # noqa: PLC0415 (scripts/bench)

        from src.config.tree import load_tree  # noqa: PLC0415
        from src.robot.drivers.ur.arm import URRobotArm  # noqa: PLC0415
        from src.robot.drivers.ur.tool_frame import tool_frame_matrix  # noqa: PLC0415

        tree = load_tree(None, root=str(config_dir))
        if not tree.ok:
            raise RuntimeError(f"the bank's config tree does not load: {tree.error}")
        self.tree = tree
        self.robot = tree.robot
        home = self.robot.home_joint_positions
        if home is None:
            raise RuntimeError("the bank's cell declares no robot.home_joint_positions for its arm to stand at")
        frame = self.robot.gripper.tool_frame
        self.tool = np.asarray(tool_frame_matrix(frame.offset_mm, frame.rotation_quat_xyzw), dtype=np.float64)
        self.state = instant_ur.ControllerState(
            model=str(self.robot.ur.model).lower(), joints=[float(v) for v in home],
            tcp_offset=instant_ur.ur_pose(self.tool) if frame.source == "polyscope" else [0.0] * 6)
        instant_ur.install(self.state)
        self.arm = URRobotArm(self.robot)
        self.arm.connect()
        self.preflight: Any = self.arm._preflight  # noqa: SLF001
        self.model = str(self.robot.ur.model).lower()
        self._services: dict[bytes, Any] = {}
        #: When the last look of the pick :meth:`world` built was taken, on the recorded frames' clock.
        self.last_look_s = _STAMP_S
        unknown = sorted(set(off) - set(LEVERS))
        if unknown:
            raise ValueError(f"no lever {unknown} to switch off; the bank knows {LEVERS}")
        self.off = tuple(off)
        if "ground" in self.off:
            from src.robot.safety.planning import support_surfaces  # noqa: PLC0415

            support_surfaces._ground_round = lambda *args, **kwargs: None  # noqa: SLF001
        if "fine_world" in self.off:
            from src.robot.execution import camera_world_wiring  # noqa: PLC0415
            from src.robot.grasping.generation import scene_obstacles, support_footprint  # noqa: PLC0415

            camera_world_wiring.FINE_REACH_MM = 0.0
            scene_obstacles.FINE_REACH_MM = 0.0
            support_footprint.SEEN_POINTS_CELL_MM = 4.0

    # -------------------------------------------------------------- the cell's own parts, per camera matrix
    def _service(self, intrinsics: np.ndarray) -> Any:
        key = np.asarray(intrinsics, dtype=np.float64).tobytes()
        if key not in self._services:
            from src.robot.execution.autonomous_grasp.service import AutonomousGraspService  # noqa: PLC0415
            from src.robot.grasping.calculator_factory import build_calculator  # noqa: PLC0415

            gripper = self.robot.gripper
            calculator = build_calculator(self.robot, data_dir=self.tree.root, camera_matrix=intrinsics,
                                          min_grip_width_mm=gripper.min_width_mm,
                                          max_grip_width_mm=gripper.max_width_mm)
            resolver = _RecordedResolver()
            service = AutonomousGraspService.from_robot_config(
                self.robot, calculator=calculator, perception=_NoCamera(), arm=self.arm, frame_resolver=resolver)
            if "parts_beside" in self.off:
                service.runtime.orchestrator.name_neighbours = False
            self._services[key] = (service, resolver)
        return self._services[key]

    def world(self, pick: RecordedPick) -> Any:
        """The camera world of every look of ``pick``, built as the cell builds it, the pick's frames held."""
        from src.robot.execution.camera_world_wiring import _live_planner_world  # noqa: PLC0415
        from src.robot.safety.planning.live_world import CameraView, DepthSnapshot  # noqa: PLC0415

        looks = pick.looks
        state = {"index": 0}

        class _Wrist:
            def grab_surface_depth(self) -> DepthSnapshot:
                look = looks[min(state["index"], len(looks) - 1)]
                return DepthSnapshot(depth_mm=look.surface_depth_mm, intrinsics=look.intrinsics,
                                     timestamp=_STAMP_S + state["index"], tool_to_base_mm=look.tool_to_base)

        world = _live_planner_world(self.robot, [CameraView(name="EIH_Cam", depth_source=_Wrist(),
                                                            camera_to_tool=looks[0].camera_to_tool)])
        if len(looks) > 1:
            world.hold_pick_views()
        for index, look in enumerate(looks):
            state["index"] = index
            world.world_for(self_envelope=self.envelope(look.joints_rad), now=_STAMP_S + index + _ASKED_AFTER_S,
                            near_point_mm=list(np.median(pick.target_cloud_mm, axis=0)))
        self.last_look_s = _STAMP_S + len(looks) - 1
        return world

    def envelope(self, joints_rad: Any) -> Any:
        from src.robot.safety.planning.self_envelope import self_envelope  # noqa: PLC0415

        return self_envelope(self.preflight, self.arm, list(joints_rad))

    # -------------------------------------------------------------- the grasps the pick would take
    def frame_of(self, pick: RecordedPick, look: Look) -> Any:
        from src.geometry import Frame, Pose  # noqa: PLC0415
        from src.robot.grasping.types.perception import PerceptionFrame  # noqa: PLC0415

        segs = tuple(_Seg(mask=m.astype(np.uint8), label=lab) for m, lab in zip(look.masks, look.labels))
        if look.target is None:
            # No target the detector named on this look: its fused cloud stands for it, beside what was named.
            segs = segs + (_Seg(mask=pick.target_mask(look).astype(np.uint8), label="target"),)
        return PerceptionFrame(depth_map=look.depth_mm, surface_depth_map=look.surface_depth_mm,
                               intrinsics=look.intrinsics, segmentations=segs, rgb=look.rgb,
                               timestamp=_STAMP_S + len(pick.looks) - 1,
                               tool_pose=Pose.from_matrix(look.tool_to_base, frame=Frame.BASE))

    def grasps(self, pick: RecordedPick, world: Any) -> "tuple[Any, Any, Any]":
        """The target's grasps on the pick's last look as the cell's orchestrator ranks them, the frame they were
        ranked on, and the orchestrator; the world then holds what the cell offers it for the attempt: the target's
        fused cloud kept out, every other named part and every part beside named by its points."""
        look = pick.looks[-1]
        service, resolver = self._service(look.intrinsics)
        orchestrator = service.runtime.orchestrator
        frame = self.frame_of(pick, look)
        resolver.by_frame[id(frame)] = look.camera_to_base
        self.arm.set_live_planner_world(world)
        self.state.teleport(list(look.joints_rad))
        orchestrator.target_label = pick.target_label
        orchestrator.target_labels = ()
        beside = orchestrator._with_the_parts_beside(frame)  # noqa: SLF001 (what every acquired frame carries)
        if beside is not frame:
            resolver.by_frame[id(beside)] = look.camera_to_base
        result, index, _ = orchestrator._best_result_over_segmentations(beside)  # noqa: SLF001
        from src.robot.grasping.loop.pick_loop import _named_points_base  # noqa: PLC0415

        named = _named_points_base(beside, tuple(beside.segmentations), index, look.camera_to_base)
        # The part the orchestrator ranked is the one held out, as the cell holds out the part it grasps: where many
        # parts share the target's word it may take another than the one the pick recorded, and that one's own points
        # stood in its world as obstacles (2026-10-13). The recorded target's cloud, fused over the looks, where it is
        # the one taken.
        recorded = look.target if look.target is not None else len(look.masks)
        held = pick.target_cloud_mm
        self.took_the_recorded_target = index is None or index == recorded
        if not self.took_the_recorded_target:
            from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415

            mask = np.asarray(beside.segmentations[index].mask).astype(bool)
            held = to_base_mm(mask, look.surface_depth_mm, look.intrinsics, look.camera_to_base)
        world.offer_segmentation(target_points_base_mm=held, target_label=pick.target_label,
                                 hold=True, timestamp=_STAMP_S + len(pick.looks) - 1, named_points_base_mm=named)
        return result, beside, orchestrator

    # -------------------------------------------------------------- the guard's word on each grasp
    def _flange(self, tcp: np.ndarray) -> np.ndarray:
        return tcp @ np.linalg.inv(self.tool)

    def _nearest(self, tcp: np.ndarray, seed: np.ndarray) -> "np.ndarray | None":
        from src.robot.safety._ur_ik import ur_flange_ik  # noqa: PLC0415

        solutions = ur_flange_ik(self.model, self._flange(tcp), q6_if_singular=float(seed[-1]))
        if not solutions:
            return None
        turned = [seed + (np.asarray(s) - seed + np.pi) % (2.0 * np.pi) - np.pi for s in solutions]
        return min(turned, key=lambda q: float(np.sum((q - seed) ** 2)))

    def _line(self, start: Any, goal: Any, seed: np.ndarray) -> "tuple[Any, str]":
        """The line from ``start`` to ``goal`` (TCP poses) as the path gate's samples, each solved nearest the one
        before, or why it cannot be solved."""
        from src.robot.safety.path_samples import PathSamples, line_samples  # noqa: PLC0415

        radii = self.preflight.joint_radii_mm(self.arm)
        step = float(self.preflight.path_step_mm or 5.0)
        samples = line_samples(start, goal, reach_mm=max(radii), max_step_mm=step)
        configs, q = [], seed
        for index, pose in enumerate(samples.poses):
            solved = self._nearest(np.asarray(pose.to_matrix(), dtype=np.float64), q)
            if solved is None:
                return None, f"sample {index + 1} of {len(samples.poses)} has no inverse kinematics"
            q = solved
            configs.append(tuple(float(v) for v in q))
        return PathSamples(configs=tuple(configs), step_bound_mm=samples.step_bound_mm), ""

    def judge(self, world: Any, grasp: Any, policy: Any, seed: Any) -> dict[str, Any]:
        """The guard's word on ``grasp``: its line down from the standoff and its lift, each sample judged by the cell's
        path gate in the world of this grasp, the space between its jaws left out."""
        waypoints = policy._build_waypoints(grasp)  # noqa: SLF001 (the poses the cell would command)
        standoff, target, top = waypoints[0], waypoints[policy.approach_steps - 1], waypoints[-1]
        tcp = np.asarray(target.to_matrix(), dtype=np.float64)
        # A grasp planned for jaws that open only as needed (lever G, a Robotiq cell) is judged with the fingers where
        # the cell opens them, as the execution tells the guard (``execution_policy._fingers_as_planned``).
        from src.robot.grasping.motion.execution_policy import _fingers_as_planned  # noqa: PLC0415

        with _fingers_as_planned(self.arm, grasp):
            return self._judged(world, waypoints, standoff, target, top, tcp, seed)

    def _judged(self, world: Any, waypoints: Any, standoff: Any, target: Any, top: Any, tcp: np.ndarray,
                seed: Any) -> dict[str, Any]:
        """:meth:`judge` once the guard holds the fingers where the grasp opens them."""
        from src.robot.core import MotionCommand  # noqa: PLC0415
        from src.robot.safety.planning.live_world import _guard_boxes  # noqa: PLC0415
        from src.robot.safety.planning.self_envelope import goal_keep_out  # noqa: PLC0415

        at = self._nearest(np.asarray(standoff.to_matrix(), dtype=np.float64), np.asarray(seed, dtype=np.float64))
        if at is None:
            return {"verdict": "no IK at the standoff"}
        down, why = self._line(standoff, target, at)
        if down is None:
            return {"verdict": f"the line down: {why}"}
        # The arm stands at the last look while the grasp is judged ahead, as on the cell: its body there is what the
        # frames of the world are filtered by.
        snapshot = world.world_for(self_envelope=self.envelope(seed),
                                   near_point_mm=[float(v) for v in tcp[:3, 3]],
                                   now=self.last_look_s + _ASKED_AFTER_S,
                                   goal_keep_out=goal_keep_out(self.preflight, self.arm, tcp))
        boxes = _guard_boxes(snapshot.perceived)
        self.preflight.set_perceived_obstacles(boxes)
        refused = self.preflight.gate_joint_path(down, arm=self.arm, command=MotionCommand.MOVE_TO)
        if refused is not None:
            return {"verdict": "line down refused", "why": str(refused.message), "boxes": len(boxes)}
        lift, why = self._line(target, top, np.asarray(down.configs[-1]))
        if lift is None:
            return {"verdict": f"the lift: {why}"}
        refused = self.preflight.gate_joint_path(lift, arm=self.arm, command=MotionCommand.MOVE_TO)
        if refused is not None:
            return {"verdict": "lift refused", "why": str(refused.message), "boxes": len(boxes)}
        return {"verdict": "admitted", "boxes": len(boxes)}


def _cause(message: str) -> str:
    """A refusal's kind, read off its sentence: what met what."""
    text = message.lower()
    pair = re.search(r"(\w+)\|(\w+)", message)
    if "seen_s" in text or "support" in text or "bench" in text:
        return "a finger or the hand at the support"
    if pair and "fixture:seen_" in text:
        return f"{pair.group(1)} at a box the camera saw"
    if "joint" in text and "limit" in text:
        return "a joint limit"
    if "cable" in text:
        return "the cable window"
    if pair:
        return f"{pair.group(1)} at {pair.group(2)}"
    return message[:80]


def replay(cell: ReplayCell, path: "str | Path") -> dict[str, Any]:
    """One recorded pick judged again: the record :mod:`run` writes."""
    started = time.perf_counter()
    pick = load_pick(path)
    record: dict[str, Any] = {"pick": pick.name, "day": pick.day, "looks": len(pick.looks),
                              "target": pick.target_label, "segmented": bool(pick.looks and pick.looks[-1].masks)}
    if not pick.looks or pick.target_cloud_mm.shape[0] < 50:
        record["skipped"] = "no target was recorded"
        return record
    world = cell.world(pick)
    result, frame, orchestrator = cell.grasps(pick, world)
    candidates = tuple(getattr(result, "candidates", ()) or ())[:MAX_TRIES] if result is not None else ()
    record["parts_beside"] = len(getattr(frame, "parts_beside", ()) or ())
    record["recorded_target"] = bool(getattr(cell, "took_the_recorded_target", True))
    record["offered"] = len(candidates)
    if not candidates:
        reasons = getattr(result, "reasons", ()) if result is not None else ()
        record["no_grasp"] = ", ".join(str(getattr(r, "value", r)) for r in reasons) or "no result"
        record["said"] = str((getattr(result, "telemetry", {}) or {}).get("no_grasp_said", ""))[:200] if result else ""
    verdicts = []
    seed = pick.looks[-1].joints_rad
    for grasp in candidates:
        try:
            verdicts.append(cell.judge(world, grasp, orchestrator.policy, seed))
        except Exception as exc:  # noqa: BLE001 (a grasp the bank cannot judge is said, never a crash of the run)
            verdicts.append({"verdict": f"not judged: {type(exc).__name__}: {exc}"[:160]})
    admitted = [i for i, v in enumerate(verdicts) if v["verdict"] == "admitted"]
    record["admitted"] = len(admitted)
    # Which stage found each grasp (``_support_footprint_stage``: SFE's prism or the part's body), and how each fared.
    by_stage: dict[str, list[int]] = {}
    for grasp, verdict in zip(candidates, verdicts):
        components = (getattr(grasp, "metadata", {}) or {}).get("score_components") or {}
        stage = str((components.get("support_footprint") or {}).get("stage", "other"))
        tally = by_stage.setdefault(stage, [0, 0])
        tally[0] += verdict["verdict"] == "admitted"
        tally[1] += 1
    record["by_stage"] = by_stage
    record["first_admitted"] = admitted[0] + 1 if admitted else None
    causes: dict[str, int] = {}
    for verdict in verdicts:
        if verdict["verdict"] != "admitted":
            kind = _cause(verdict.get("why", verdict["verdict"]))
            causes[kind] = causes.get(kind, 0) + 1
    record["refused"] = causes
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record
