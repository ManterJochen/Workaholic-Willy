"""The hard-scene bench: the owner's own cell picks in scenes nobody wants, every pick a whole pick of its real stack.

    python scripts/bench/run_bench.py --scenes all --work logs/bench/<run>
    python scripts/bench/run_bench.py --scenes bin,clutter_L_15 --looks first

What runs is the owner's cell as its Monday tree builds it (``--tree``, copied, never written): the UR driver with its
judged lines and cuRobo routes, the exact guard at the cell's distances, the camera world, the grasp calculator, the
recovery (rescan, next target, push, clear a blocker) and the toggle Hand-E on tool DO0. Three things stand in, and
nothing else:

* **The controller** (``instant_ur.py``): a judged ``moveJ`` or ``moveL`` is taken at once, so a pick takes seconds of
  computing, not the minutes a real arm, or URSim, takes to drive it.
* **The camera** (``scripts/ursim/_mat_scene.py``): every grab renders the scene from where the wrist camera stands,
  the owner's mat cast whole, the parts and bins as the scene places them; the detector grounds the target's mask.
* **The parts**: they do not fall or slide. A part the jaws close on is carried with the TCP, a push moves its part as
  far as the plan says. Whether a grasp holds is a question for the physics (MuJoCo or Isaac), not for this bench.

The bench's layer (``robot.bench.yaml``) puts the owner's 2026-10-05 numbers over the tree: 3 mm for the guard and the
lines, 8 mm round a box the camera saw, the cable window, and recovery with the push allowed. A pick counts as a success
where the target is in the jaws when it ends. Each scene writes its looks (``views/<scene>.npz``) with one debug picture
per look (the detections, their masks, the ranked grasps), and the run a ``bench_result.json`` and a ``report.html``.

While the owner games on this box (2026-10-05), run it only when they say so: the planner takes the GPU.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import re
import shutil
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for path in (str(_REPO), str(_HERE), str(_HERE.parent / "ursim")):
    if path not in sys.path:
        sys.path.insert(0, path)

import _mat_scene as ms  # noqa: E402
import instant_ur  # noqa: E402
from scenes import SCENES, scene_names  # noqa: E402

DEFAULT_TREE = Path(r"D:/dev/willy_handover/2026-10-02/final/owner_monday/config")
DEFAULT_CALIBRATION = Path(r"D:/helper_cell/calibration/real/eih_EIH_Cam.json")
DEFAULT_CROP = _REPO / "tests" / "data" / "cell_2026_10_01" / "p1_pick-63e1b3dac29f.npz"
BENCH_LAYER = "bench"
PROMPT = "part"

_LAYER = {
    "robot": {
        "ur": {"ip": "127.0.0.1"},
        "safety": {
            "joint_limits": {"within_half_turn_of_home": True},
            "self_collision": {"min_distance_mm": 3.0, "perceived_min_distance_mm": 3.0},
            "planned_motion": {"line_clearance_mm": 3.0},
            "planning_world": {"perceived": {"margin_mm": 8.0}},
        },
        "grasping": {
            "recovery": {"enabled": True, "allowed_actions": ["rescan", "next_target", "nudge_target"],
                         "per_action_budget": [["rescan", 1]], "critical_parts": False},
        },
    }
}


def copy_the_tree(tree: Path, work: Path, calibration: Optional[Path], *, no_planner: bool = False,
                  ip: str = "127.0.0.1", margin_mm: Optional[float] = None) -> Path:
    """``tree`` copied to ``<work>/config`` with the bench's layer, and the camera layer naming ``calibration``.
    ``no_planner`` runs the arm on its controller's IK, no cuRobo: a wiring check that needs no GPU, not a result.
    ``ip`` is the controller's address, which nothing dials (the instant controller answers every address) and which
    names the cell's lock: two benches at once need two."""
    copied = work / "config"
    shutil.copytree(tree, copied)
    layer = json.loads(json.dumps(_LAYER))
    layer["robot"]["ur"]["ip"] = str(ip)
    if margin_mm is not None:
        # An A/B of how far the camera world grows its boxes past the points (``--margin``); the owner's 8 otherwise.
        layer["robot"]["safety"]["planning_world"]["perceived"]["margin_mm"] = float(margin_mm)
    if no_planner:
        layer["robot"]["ur"]["motion_planner"] = "ik"
    text = ("# The bench's layer (scripts/bench/run_bench.py): the owner's numbers of 2026-10-05 over the tree, and the\n"
            "# instant controller's address. Nothing else changes.\n") + yaml.safe_dump(layer, sort_keys=False)
    (copied / "robot" / f"robot.{BENCH_LAYER}.yaml").write_text(text, encoding="utf-8")
    if calibration is not None:
        base = yaml.safe_load((copied / "camera" / "cam.yaml").read_text(encoding="utf-8")) or {}
        cameras = base.get("cameras") or {}
        primary = cameras.get("primary_rig_id")
        rigs = cameras.get("rigs") or []
        for rig in rigs:
            if rig.get("rig_id") == primary and isinstance(rig.get("extrinsics"), dict):
                rig["extrinsics"]["artifact_path"] = calibration.resolve().as_posix()
        (copied / "camera" / f"cam.{BENCH_LAYER}.yaml").write_text(
            "# The bench's camera layer: the primary rig's calibration where this box holds it.\n"
            + yaml.safe_dump({"cameras": {"rigs": rigs}}, sort_keys=False), encoding="utf-8")
    return copied


def _tcp_offset(robot: Any) -> list[float]:
    from src.robot.drivers.ur.tool_frame import tool_frame_matrix  # noqa: PLC0415

    frame = robot.gripper.tool_frame
    ox, oy, oz = (float(v) for v in frame.offset_mm)
    qx, qy, qz, qw = (float(v) for v in frame.rotation_quat_xyzw)
    matrix = tool_frame_matrix((ox, oy, oz), (qx, qy, qz, qw))
    return instant_ur.ur_pose(matrix)


def _report_dict(report: Any) -> dict[str, Any]:
    pick = getattr(report, "pick_report", None)
    attempts = []
    for attempt in getattr(pick, "attempts", ()) or ():
        attempts.append({
            "index": getattr(attempt, "attempt_index", None), "action": getattr(attempt, "action", None),
            "reasons": [str(getattr(r, "value", r)) for r in (getattr(attempt, "reasons", ()) or ())],
            "push": getattr(attempt, "push", None), "blocker": getattr(attempt, "blocker", None),
            "motion_status": str(getattr(attempt, "motion_status", None)),
            "motion_message": str(getattr(attempt, "motion_message", "") or "")[:400],
            "tries": [{k: (str(v)[:300] if isinstance(v, str) else v) for k, v in dict(t).items()}
                      for t in (getattr(attempt, "tries", ()) or ())],
        })
    telemetry = getattr(report, "telemetry", {}) or {}
    trail = {k: telemetry.get(k) for k in ("reasons", "low_level_outcome", "recovery_trail_actions",
                                          "recovery_trail_terminal_reason", "pushes") if k in telemetry}
    try:
        summary = report.failure_summary()
    except Exception as exc:  # noqa: BLE001
        summary = f"({type(exc).__name__}: {exc})"
    return {
        "outcome": str(getattr(getattr(report, "outcome", None), "value", getattr(report, "outcome", None))),
        "pick_outcome": str(getattr(getattr(pick, "outcome", None), "value", None)),
        "failure_summary": summary,
        "needs_person": str(getattr(report, "needs_person", "") or ""),
        "attempts": attempts,
        "calculator": {k: telemetry.get(k) for k in ("no_grasp_said", "rejected_collision", "rejected_table",
                                                      "support_footprint_refused") if k in telemetry},
        "trail": json.loads(json.dumps(trail, default=str)),
    }


#: The log lines that say why a pick went the way it did, kept per pick (the pick loop's, the calculator's, recovery's).
_TELLING = ("where the carried lift starts", "no grasp", "clear the blocker", "no push of the part", "not taken away", "judged from where", "try ",
            "pushing the part", "the push came", "refused", "no collision-free plan", "Generated", "skips",
            "taking away", "next_target", "rescan", "only excluded", "every grasp")


class _Telling(logging.Handler):
    """Keeps the telling log lines of one pick (:data:`_TELLING`), at most ``limit``; a line naming a box the camera
    saw (``|fixture:seen_..``) is followed by that box as the guard held it (``boxes``: the guard's boxes now)."""

    def __init__(self, boxes: Any = None, limit: int = 120) -> None:
        super().__init__(level=logging.INFO)
        self.limit = limit
        self.boxes = boxes
        self.lines: list[str] = []
        self.named: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 (a line that cannot render is not kept)
            return
        if len(self.lines) >= self.limit:
            return
        if any(word in message for word in _TELLING) or "|fixture:" in message:
            self.lines.append(f"{record.name.rsplit('.', 1)[-1]}: {message[:400]}")
        for name in re.findall(r"\|fixture:(seen_[A-Za-z0-9_]+)", message):
            if name in self.named or self.boxes is None:
                continue
            self.named.add(name)
            box = next((b for b in self.boxes() if getattr(b, "name", "") == name), None)
            if box is not None:
                self.lines.append(f"box {name}: {_box_said(box)}")


def _box_said(box: Any) -> str:
    """A guard box in one line: its centre, its turned sizes and turn, and the box square with BASE round it."""
    centre = np.round(np.asarray(box.center_mm, dtype=np.float64), 1).tolist()
    outer = np.round(2.0 * np.asarray(box.half_extents_mm, dtype=np.float64), 1).tolist()
    turned = getattr(box, "turned", None)
    if turned is None:
        return f"centre {centre} size {outer}"
    sizes = np.round(2.0 * np.asarray(turned.half_extents_mm, dtype=np.float64), 1).tolist()
    return (f"centre {centre} turned size {sizes} yaw {np.degrees(float(turned.yaw_rad)):.1f} deg, square with BASE "
            f"{outer}{'' if getattr(box, 'note', '') in ('', None) else ' (' + str(box.note) + ')'}")


#: How many failed picks in a row end a clearing, as they end the owner's task (``TaskPlan.max_failed_in_a_row``).
FAILED_IN_A_ROW = 3

class Bench:
    """The cell on the instant controller, the scene, and what each pick came to."""

    def __init__(self, args: argparse.Namespace) -> None:
        from src.config.tree import load_tree  # noqa: PLC0415

        self.args = args
        self.work = Path(args.work)
        self.work.mkdir(parents=True, exist_ok=True)
        copied = copy_the_tree(Path(args.tree), self.work, None if args.no_calibration else Path(args.calibration),
                               no_planner=bool(args.no_planner), ip=str(args.ip),
                               margin_mm=getattr(args, "margin", None))
        chain = f"{args.profile},{BENCH_LAYER}" if args.profile else BENCH_LAYER
        self.tree = load_tree(chain, root=copied)
        robot = self.tree.robot
        self.robot = robot
        #: The cell's word on where a blocker of a clearing goes (``recovery.blocker_into_the_place``).
        self.blocker_into_the_place = bool(getattr(robot.grasping.recovery, "blocker_into_the_place", True))
        self.home = [float(v) for v in (robot.home_joint_positions or ())]
        self.state = instant_ur.ControllerState(model=str(robot.ur.model).lower(), joints=list(self.home),
                                                tcp_offset=_tcp_offset(robot))
        static = ms.StaticScene.from_crop(Path(args.crop))
        static.whole_bench = True
        self.static = static
        # The depth the stand-in camera hands the stack: the cell's D415 as its frames read it, unless a run asks for
        # the clean render (``--noise clean``).
        self.scene = ms.Scene(static, depth_model=str(getattr(args, "noise", "real")))
        self.jaws_closed = False
        self.state.on_tool_output = self._tool_output
        self.results: list[dict[str, Any]] = []
        self.backend: Any = None
        #: The boxes the camera saw that the guard holds now, as the last world handed them (for the log of a pick).
        self.guard_boxes: tuple[Any, ...] = ()
        #: The last few calls of ``build_perceived_boxes``, their arguments whole (``_write_worlds``).
        self.worlds_built: list[tuple[Any, Any]] = []

    def _tool_output(self, pin: int, level: bool, tcp: np.ndarray) -> None:
        """The owner's valve: every change of the tool output moves the jaws (2026-09-28)."""
        if pin != int(self.robot.gripper.jaw_io.close_output_pin):
            return
        self.jaws_closed = not self.jaws_closed
        self.scene.jaws_changed(self.jaws_closed, tcp)

    def looks(self) -> list[Any]:
        from src.robot.core import JointPositions  # noqa: PLC0415

        rows = [tuple(row) for row in (self.robot.look_joint_positions_deg or ())]
        if self.args.looks == "first":
            rows = rows[:1]
        return [JointPositions.deg(*row) for row in rows]

    def run(self, names: list[str]) -> dict[str, Any]:
        from src.camera.orchestration import camera as camera_module  # noqa: PLC0415
        from src.models import perception_spec as spec_module  # noqa: PLC0415
        from src.robot.execution.cell import Cell  # noqa: PLC0415
        from src.robot.grasping.motion.grasp_motion import GraspMotion  # noqa: PLC0415
        from src.robot.grasping.recovery import push_motion  # noqa: PLC0415
        from src.robot.grippers.jaw_io import JawIOGripper  # noqa: PLC0415

        from src.calibration.rig_calibration import RigCalibration  # noqa: PLC0415

        section = self.tree.app_config.camera.cameras
        rig = next(r for r in section.rigs if r.rig_id == section.primary_rig_id)
        calibration = RigCalibration.from_config(rig.rig_id, getattr(rig, "extrinsics", None))
        cam_to_tool = np.asarray(calibration.camera_to_tool().to_matrix(), dtype=np.float64)
        cameras: list[ms.StandInD415] = []

        def create_streamer(rig: Any) -> ms.StandInD415:
            camera = ms.StandInD415(rig, self.scene, camera_to_tool=cam_to_tool, tcp=lambda: self.state.tcp_mm())
            cameras.append(camera)
            return camera

        def spec_from_config(cls: Any, models: Any) -> ms.StandInSpec:
            self.backend = ms.StandInBackend(cameras[-1])
            return ms.StandInSpec(self.backend)

        original_execute_push = push_motion.execute_push
        from src.robot.safety.preflight import SafetyPreflight  # noqa: PLC0415

        original_set = SafetyPreflight.set_perceived_obstacles
        bench = self
        from src.robot.drivers.ur.arm import URRobotArm  # noqa: PLC0415

        original_carried = URRobotArm.carried_line_refusal

        def carried_line_refusal(arm: Any, pose: Any, **kwargs: Any) -> Any:
            # Which pairs the planner's world refuses where the lift starts, said beside a carried lift it refuses.
            judged = original_carried(arm, pose, **kwargs)
            start = kwargs.get("start")
            planner = getattr(arm, "_curobo_ur", None)
            if judged is not None and start is not None and planner is not None:
                try:
                    said = planner.judge_joint_path([list(start[1])], name_pairs=True).render()
                except Exception as exc:  # noqa: BLE001 (a diagnosis never fails a pick)
                    said = f"({type(exc).__name__}: {exc})"
                logging.getLogger("bench").warning("refused where the carried lift starts: %s", said)
            return judged

        def set_perceived_obstacles(preflight: Any, boxes: Any) -> int:
            bench.guard_boxes = tuple(boxes)
            return original_set(preflight, boxes)

        # The last few worlds the camera world built, their inputs whole: a pick refused for cells the robot hid writes
        # them out (``worlds/<scene>.pkl``), to be built again offline (``replay_world.py``).
        from src.robot.safety.planning import live_world as live_world_module  # noqa: PLC0415

        original_build = live_world_module.build_perceived_boxes

        def build_perceived_boxes(*args: Any, **kwargs: Any) -> Any:
            bench.worlds_built.append((args, kwargs))
            del bench.worlds_built[:-4]
            return original_build(*args, **kwargs)

        def execute_push(arm: Any, plan: Any, **kwargs: Any) -> Any:
            outcome = original_execute_push(arm, plan, **kwargs)
            if outcome is not None and getattr(outcome, "plan", None) is not None and outcome.motion_started:
                self.scene.pushed(outcome.plan, tuple(outcome.legs_done), self.state.tcp_mm())
            return outcome

        before = instant_ur.install(self.state)
        # The console's own motion (the policy's 80 mm standoff) unless a run names a standoff to try.
        from src.contracts import UNSET  # noqa: PLC0415

        motion: Any = UNSET if self.args.standoff is None else GraspMotion(standoff_mm=float(self.args.standoff))
        summary: dict[str, Any] = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "scenes": {},
                                   "noise": str(getattr(self.args, "noise", "real"))}
        try:
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(camera_module, "create_streamer", create_streamer))
                stack.enter_context(mock.patch.object(spec_module.PerceptionSpec, "from_config",
                                                      classmethod(spec_from_config)))
                stack.enter_context(mock.patch.object(JawIOGripper, "_question",
                                                      lambda gripper: (lambda q: "closed" if self.jaws_closed
                                                                       else "open")))
                stack.enter_context(mock.patch.object(push_motion, "execute_push", execute_push))
                stack.enter_context(mock.patch.object(SafetyPreflight, "set_perceived_obstacles",
                                                      set_perceived_obstacles))
                stack.enter_context(mock.patch.object(URRobotArm, "carried_line_refusal", carried_line_refusal))
                stack.enter_context(mock.patch.object(live_world_module, "build_perceived_boxes", build_perceived_boxes))
                cell = Cell.from_tree(self.tree, prompt=PROMPT, motion=motion)
                t0 = time.time()
                cell.build()
                summary["build_s"] = round(time.time() - t0, 1)
                with cell.connected():
                    self.cell, self.service = cell, cell.service
                    for name in names:
                        print(f"\n=== {name} ===", flush=True)
                        try:
                            row = self.one(name)
                        except Exception as exc:  # noqa: BLE001 (a scene that raised is a failed row, said)
                            row = {"scene": name, "success": False, "raised": f"{type(exc).__name__}: {exc}",
                                   "traceback": traceback.format_exc()[-3000:]}
                        summary["scenes"][name] = row
                        print(json.dumps({k: row.get(k) for k in ("success", "seconds", "outcome", "why")},
                                         default=str), flush=True)
                        (self.work / "bench_result.json").write_text(json.dumps(summary, indent=1, default=str),
                                                                    encoding="utf-8")
        finally:
            instant_ur.remove(before)
        summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (self.work / "bench_result.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        write_report(self.work, summary)
        return summary

    def _open_and_forget(self) -> None:
        """The jaws open and the arm forgets the part it carried, as ``Robot.release`` does at a place: a part the planner
        still thought carried refused every approach of the next scene."""
        if self.jaws_closed:
            self.cell.gripper.set_closed(False)
        detach = getattr(self.cell.arm, "detach_payload", None)
        if callable(detach):
            detach()

    def _reset(self) -> None:
        self._open_and_forget()
        self.state.teleport(self.home)
        if getattr(self.service, "stopped_where_the_arm_stands", None):
            self.service.acknowledge_needs_person()

    def _pick_once(self, label: str) -> dict[str, Any]:
        """One whole pick of the service from home; what it came to, with its looks' debug pictures."""
        from src.robot.execution.pick_run import _keep_debug_images  # noqa: PLC0415
        from src.robot.execution.record_views import record_views  # noqa: PLC0415

        self._reset()
        t0 = time.time()
        telling = _Telling(boxes=lambda: self.guard_boxes)
        # The stack's loggers keep to their own files (``propagate`` off), so the handler goes on each of them.
        loggers = [logging.getLogger()] + [logger for logger in logging.root.manager.loggerDict.values()
                                           if isinstance(logger, logging.Logger) and not logger.propagate]
        for logger in loggers:
            logger.addHandler(telling)
        try:
            report = self.service.pick(look=self.looks())
        finally:
            for logger in loggers:
                logger.removeHandler(telling)
        held = self.scene.held()
        row: dict[str, Any] = {"seconds": round(time.time() - t0, 1),
                               "held": None if held is None else held.name, **_report_dict(report)}
        orchestrator = self.service.runtime.orchestrator
        row["pushes"] = [p.to_dict() for p in getattr(orchestrator, "pushes", ()) or ()]
        row["blockers"] = [b.to_dict() if hasattr(b, "to_dict") else repr(b)
                           for b in getattr(orchestrator, "blockers", ()) or ()]
        row["scene_events"] = [e for e in self.scene.events if e.get("t", 0) >= t0]
        row["log"] = telling.lines
        if "hid from the cameras" in str(row.get("failure_summary") or ""):
            row["worlds"] = self._write_worlds(label)
        # The boxes the camera saw that the guard held last, by name and size: where the world spends its slots.
        row["guard_boxes"] = [f"{getattr(box, 'name', '')}: {_box_said(box)}" for box in self.guard_boxes]
        looked = getattr(self.service, "looked_around", None)
        views = getattr(looked, "views", None)
        if isinstance(views, tuple) and views:
            try:
                written = record_views(views, target_cloud_base_mm=getattr(getattr(looked, "judged", None),
                                                                            "target_cloud_base_mm", None),
                                       directory=self.work / "views", name=label)
                if written is not None:
                    ranked = getattr(getattr(self.service, "ranked_looked", None), "judged", None)
                    row["pictures"] = [str(p.relative_to(self.work)) for p in
                                       _keep_debug_images(views, getattr(looked, "judged", None), Path(written),
                                                          ranked=ranked)]
            except Exception as exc:  # noqa: BLE001 (a picture never fails a scene)
                row["pictures_error"] = f"{type(exc).__name__}: {exc}"
        return row

    def _write_worlds(self, label: str) -> str:
        """The last worlds' inputs, pickled to ``worlds/<label>.pkl``; where they went, or why they did not."""
        import pickle  # noqa: PLC0415

        path = self.work / "worlds" / f"{label}.pkl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(list(self.worlds_built), handle, protocol=pickle.HIGHEST_PROTOCOL)
            return str(path.relative_to(self.work))
        except Exception as exc:  # noqa: BLE001 (a diagnosis never fails a scene)
            return f"not written: {type(exc).__name__}: {exc}"

    def _take_away(self, name: str) -> None:
        """The part the jaws hold leaves the scene, as a part set down elsewhere does, and the jaws open on nothing."""
        with self.scene._lock:  # noqa: SLF001 (the scene's own lock)
            self.scene.parts.pop(name, None)
            self.scene.version += 1
        self._open_and_forget()

    def one(self, name: str) -> dict[str, Any]:
        spec = SCENES[name]
        self._reset()
        parts = spec.build(self.static)
        self.scene.set_parts(parts, target=spec.target)
        self.scene.keep_height_on_push = spec.keep_height_on_push
        self.backend.all_parts = spec.mode == "clear"
        # A clearing takes every part, as the owner's task "alle Objekte ... in die gelbe Kiste" does: a blocker is the
        # part a pick takes and leaves the scene as a picked part does (``recovery.blocker_into_the_place``).
        into = spec.mode == "clear" and self.blocker_into_the_place
        self.service.start_campaign(blocker_is_the_pick=into)
        base = {"scene": name, "family": spec.family, "note": spec.note, "possible": spec.possible,
                "impossible": spec.impossible, "mode": spec.mode}
        if spec.mode != "clear":
            row = {**base, **self._pick_once(name)}
            row["success"] = row.get("held") == spec.target
            row["why"] = "" if row["success"] else (row.get("failure_summary") or row.get("outcome"))
            self.results.append(row)
            return row
        movable = [p.name for p in parts if not p.fixed]
        picks: list[dict[str, Any]] = []
        failed_in_a_row = 0
        t0 = time.time()
        # A clearing ends where the task's does: three failed picks in a row (``TaskPlan.max_failed_in_a_row``).
        while failed_in_a_row < FAILED_IN_A_ROW and len(picks) < int(len(movable) * 1.5) + 2:
            left = [n for n, p in self.scene.parts.items() if not p.fixed]
            if not left:
                break
            pick = self._pick_once(f"{name}_pick{len(picks):02d}")
            pick["left_before"] = len(left)
            picks.append(pick)
            print(f"  pick {len(picks)}: {pick.get('held') or pick.get('outcome')} ({pick['seconds']} s, "
                  f"{len(left)} left)", flush=True)
            if pick.get("held"):
                self._take_away(str(pick["held"]))
                failed_in_a_row = 0
            else:
                self._open_and_forget()
                failed_in_a_row += 1
            if getattr(self.service, "stopped_where_the_arm_stands", None):
                self.service.acknowledge_needs_person()
                self.service.start_campaign(blocker_is_the_pick=into)
        picked = [p["held"] for p in picks if p.get("held")]
        row = {**base, "parts": len(movable), "picked": len(picked), "picked_names": picked,
               "left": [n for n, p in self.scene.parts.items() if not p.fixed], "seconds": round(time.time() - t0, 1),
               "picks": picks}
        row["success"] = len(picked) >= 0.8 * len(movable)
        row["outcome"] = f"{len(picked)}/{len(movable)} picked"
        last_fail = next((p for p in reversed(picks) if not p.get("held")), None)
        row["why"] = "" if row["success"] else str((last_fail or {}).get("failure_summary") or "")
        row["pictures"] = [pic for p in picks for pic in p.get("pictures", [])][:12]
        self.results.append(row)
        return row


def write_report(work: Path, summary: dict[str, Any]) -> Path:
    """``report.html``: the rate over the possible scenes, per family, and one card per scene with its pictures."""
    rows = list(summary.get("scenes", {}).values())
    possible = [r for r in rows if r.get("possible", True)]
    done = [r for r in possible if r.get("success")]
    families: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        families.setdefault(str(r.get("family", "?")), []).append(r)

    def rate(items: list[dict[str, Any]]) -> str:
        items = [r for r in items if r.get("possible", True)]
        return f"{sum(1 for r in items if r.get('success'))}/{len(items)}" if items else "-"

    cards = []
    for r in rows:
        pictures = "".join(
            f'<img src="data:image/png;base64,{base64.b64encode((work / p).read_bytes()).decode("ascii")}" alt="">'
            for p in r.get("pictures", []) if (work / p).is_file())
        state = "ok" if r.get("success") else ("impossible" if not r.get("possible", True) else "fail")
        actions = ", ".join(str(a.get("action")) for a in r.get("attempts", []))
        cards.append(
            f'<section class="card {state}"><h3>{html.escape(str(r.get("scene")))} '
            f'<span>{html.escape(state)}</span></h3><p>{html.escape(str(r.get("note", "")))}</p>'
            f'<p><b>Outcome:</b> {html.escape(str(r.get("outcome")))} · <b>Attempts:</b> {html.escape(actions)} · '
            f'<b>{r.get("seconds", "?")} s</b></p>'
            + (f'<p class="why">{html.escape(str(r.get("why"))[:600])}</p>' if r.get("why") else "")
            + (f'<p class="why">{html.escape(str(r.get("raised")))}</p>' if r.get("raised") else "")
            + f'<div class="pics">{pictures}</div></section>')
    table = "".join(f"<tr><td>{html.escape(f)}</td><td>{rate(items)}</td></tr>" for f, items in families.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Grasp bench</title>
<style>body{{font:14px system-ui;margin:16px;background:#111;color:#eee}}.card{{border:1px solid #444;margin:12px 0;
padding:8px;border-radius:6px}}.ok{{border-color:#2a7}}.fail{{border-color:#c44}}.impossible{{border-color:#777}}
h3 span{{font-size:12px;opacity:.8}}.pics img{{max-width:420px;margin:4px}}.why{{color:#f99}}table td{{padding:2px 10px}}
</style></head><body><h1>Grasp bench</h1><p>{summary.get("started")} – {summary.get("finished", "running")} ·
possible scenes picked: <b>{len(done)}/{len(possible)}</b></p><table>{table}</table>{"".join(cards)}</body></html>"""
    out = work / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", default=str(DEFAULT_TREE), help="the cell's config folder; copied, never written")
    ap.add_argument("--profile", default="monday", help="the tree's own chain; the bench's layer is added after it")
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    ap.add_argument("--no-calibration", action="store_true", help="keep the tree's own calibration path")
    ap.add_argument("--crop", default=str(DEFAULT_CROP))
    ap.add_argument("--scenes", default="all", help="all, a family (bin, clutter, tray, shapes) or names")
    ap.add_argument("--looks", choices=("all", "first"), default="all")
    ap.add_argument("--ip", default="127.0.0.1", help="the cell's lock key; give a second bench run another (127.0.0.2)")
    ap.add_argument("--standoff", type=float, default=None,
                    help="the approach's standoff in mm; the console's own (the policy's 80 mm) where not given")
    ap.add_argument("--no-planner", action="store_true",
                    help="a wiring check with no cuRobo (no GPU): the arm on IK alone; its rates mean nothing")
    ap.add_argument("--noise", choices=ms.DEPTH_MODELS, default="real",
                    help="the camera's depth: real, the cell's D415 as its frames of 2026-10-07 read it (noise, holes, "
                         "shadows, blurred edges); harsh, worse than that, to stress the stack; clean, the render as it "
                         "was")
    ap.add_argument("--margin", type=float, default=None,
                    help="an A/B of how far the camera world grows its boxes past the points "
                         "(safety.planning_world.perceived.margin_mm), millimetres; the owner's 8 where not given")
    ap.add_argument("--slack", type=float, default=None,
                    help="an A/B of the calculator's slack past the guard's distance (scene_obstacles.WHOLE_SLACK_MM), "
                         "millimetres; the shipped value where not given")
    ap.add_argument("--work", default=str(_REPO / "logs" / "bench" / time.strftime("%Y%m%d_%H%M%S")))
    args = ap.parse_args(argv)
    if args.slack is not None:
        from src.robot.grasping.generation import scene_obstacles  # noqa: PLC0415

        scene_obstacles.WHOLE_SLACK_MM = float(args.slack)
        print(f"the calculator plans the hand {args.slack:g} mm past the guard's distance (--slack)", flush=True)
    names = scene_names(args.scenes)
    bench = Bench(args)
    summary = bench.run(names)
    rows = list(summary["scenes"].values())
    possible = [r for r in rows if r.get("possible", True)]
    print(f"\npossible scenes picked: {sum(1 for r in possible if r.get('success'))}/{len(possible)}; "
          f"report {Path(args.work) / 'report.html'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
