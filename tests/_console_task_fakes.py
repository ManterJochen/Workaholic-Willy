"""Doubles for the console's run tests: the console_dummy tree, and a scripted cell installed in a real ``Console``.

Two cells, for two kinds of question.

* **console_dummy** (:func:`dummy_tree`): the shipped tree with its arm and hand swapped for the dummies, plus a
  profile layer ``consoletest`` that teaches two poses (``drop_left`` "Ablage links", the default place, and ``park``
  "Parkposition"). Built with ``rehearse=true`` it is the console end to end at a desk: the real service, the real pick
  loop on the rehearsal scene, the dummy arm. :class:`EmptyingScene` empties that scene after a number of frames, so
  "until empty" ends there.
* **a scripted cell** (:class:`ScriptedCell`): the owner's cell in miniature, through the real console and the real
  ``run_task``. The arm is :class:`ConsoleArm`, the task tests' ``TaskArm`` (every motion on one log, a planner that judges its
  paths, a carried part modelled) with what the console reads besides: a connection, the halt latch with a real
  ``HaltState``, the controller's fields, the planner's state. The hand is the real ``JawIOGripper`` single_toggle on
  tool DO0 over a recording I/O the latch refuses. The service is :class:`ConsoleService`: each ``pick()`` plays the next
  :class:`Scripted` pick as the pick loop would, its progress events, its overlay, a push, an empty look or a motion the
  halt refused, on the real arm and hand, and advances a :class:`VirtualClock` by the pick's seconds.

Nothing here talks to a controller, a camera or a GPU.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from src.config.loader import active_profile, reload_config, set_active_profile
from src.robot.core import JointPositions, MotionStatus, RobotStatus
from src.robot.core.arm_capabilities import ArmHalted, HaltState, PayloadModel, halted_refusal
from src.robot.core.errors import RobotConnectionError
from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
from src.robot.grasping.loop.pick_loop import PickOutcome
from src.robot.grasping.loop.progress import PickProgress, PickStage
from src.robot.grippers.jaw_io import JawIOGripper
from tests._task_fakes import (
    GRASP_Z_MM,
    PARK_JOINTS,
    PART_XY,
    PLACE_JOINTS,
    RUNNING,
    ScriptedLocator,
    TaskArm,
    TaskService,
    _report,
)
from tests.test_a_stopped_controller_moves_no_jaws import _IO

__all__ = [
    "AnswersOpen",
    "ConsoleArm",
    "ConsoleCase",
    "ConsoleService",
    "EmptyingScene",
    "LOOK_1",
    "LOOK_2",
    "POSES_DEG",
    "PROFILE",
    "Scripted",
    "ScriptedCell",
    "SightedLocator",
    "VirtualClock",
    "await_run",
    "dummy_tree",
    "event_types",
    "events_of",
    "wait_for_event",
]

_SHIPPED = Path(__file__).resolve().parents[1] / "config"

#: The profile layer the scratch trees teach their poses in.
PROFILE = "consoletest"
#: The taught poses of every scratch tree, by name: (label, joints in degrees). ``drop_left`` is the default place.
POSES_DEG: Mapping[str, tuple[str, tuple[float, ...]]] = {
    "drop_left": ("Ablage links", tuple(PLACE_JOINTS.degrees())),
    "park": ("Parkposition", tuple(PARK_JOINTS.degrees())),
}
#: Two looks a scripted wrist cell looks from.
LOOK_1 = JointPositions.deg(10.0, -80.0, -100.0, -90.0, 90.0, 0.0)
LOOK_2 = JointPositions.deg(-10.0, -80.0, -100.0, -90.0, 90.0, 0.0)


def dummy_tree(target: Path, *, gripper: str = "dummy", poses: bool = True, default_place: bool = True) -> str | None:
    """Copy the shipped tree to ``target``, its arm the dummy and its hand ``gripper`` (``dummy`` or ``none``).

    With ``poses`` a profile layer :data:`PROFILE` teaches :data:`POSES_DEG` (``drop_left`` the default place where
    ``default_place``); the profile to load it with is returned, ``None`` without poses.
    """
    shutil.copytree(_SHIPPED, target)
    robot = target / "robot" / "robot.yaml"
    text = robot.read_text(encoding="utf-8")
    text, arm_hits = re.subn(r'^(\s*)vendor:\s*"ur"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE)
    text, hand_hits = re.subn(r'^(\s*)vendor:\s*"robotiq"$', rf'\g<1>vendor: "{gripper}"', text, count=1,
                              flags=re.MULTILINE)
    assert arm_hits == 1 and hand_hits == 1, "the dummy substitution found nothing"
    robot.write_text(text, encoding="utf-8")
    if not poses:
        return None
    lines = ["robot:", "  named_poses:"]
    for name, (label, joints) in POSES_DEG.items():
        lines += [f"    {name}:", f"      joints_deg: [{', '.join(f'{v:.1f}' for v in joints)}]",
                  f'      label: "{label}"', "      screen: clear"]
    if default_place:
        lines.append("  default_place_pose: drop_left")
    (target / "robot" / f"robot.{PROFILE}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PROFILE


# ---------------------------------------------------------------------------------------------------------------------
# Waiting on runs and reading their events
# ---------------------------------------------------------------------------------------------------------------------


def events_of(console: Any, stream: str) -> list[Any]:
    """Every event of ``stream`` the console's hub holds, in order."""
    return list(console.hub.since(stream, 0)[0])


def event_types(console: Any, stream: str) -> list[str]:
    return [event.type for event in events_of(console, stream)]


def wait_for_event(console: Any, stream: str, event_type: str, *, timeout: float = 20.0) -> Any:
    """The first ``event_type`` on ``stream``, waited for."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in console.hub.since(stream, 0)[0]:
            if event.type == event_type:
                return event
        time.sleep(0.005)
    raise AssertionError(f"no {event_type} on {stream} within {timeout} s: {event_types(console, stream)}")


def await_run(console: Any, run_id: str, *, timeout: float = 30.0) -> Any:
    """The run once it ended: its ``run_finished`` is out, so its thread has let go of everything."""
    wait_for_event(console, run_id, "run_finished", timeout=timeout)
    run = console.registry.get(run_id)
    assert run is not None, f"run {run_id} is forgotten"
    return run


# ---------------------------------------------------------------------------------------------------------------------
# The rehearsal scene that empties
# ---------------------------------------------------------------------------------------------------------------------


class EmptyingScene:
    """The rehearsal scene, empty from frame ``after + 1`` on: "until empty" on console_dummy ends ``nothing_left``.

    Wraps the cell's ``RehearsalPerceptionSource`` in place (:meth:`install`): every frame it serves after ``after``
    shows the same plane with no box on it.
    """

    def __init__(self, source: Any, *, after: int) -> None:
        self.source = source
        self.after = after
        self.served = 0
        self._acquire = source.acquire

    @classmethod
    def install(cls, service: Any, *, after: int) -> "EmptyingScene":
        source = service.runtime.orchestrator.perception
        scene = cls(source, after=after)
        source.acquire = scene.acquire
        return scene

    def acquire(self) -> Any:
        frame = self._acquire()
        self.served += 1
        if self.served <= self.after:
            return frame
        depth = np.full_like(frame.depth_map, float(np.max(frame.depth_map)))
        return dataclasses.replace(frame, depth_map=depth, segmentations=(), rgb=np.zeros_like(frame.rgb))


# ---------------------------------------------------------------------------------------------------------------------
# The scripted cell: the arm
# ---------------------------------------------------------------------------------------------------------------------


class _LatchedIO(_IO):
    """The tool I/O as the arm drives it: while the arm's halt latch is set a write is refused, nothing sent (L9)."""

    def __init__(self, log: list[Any], arm: "ConsoleArm") -> None:
        super().__init__(log)
        self.arm = arm

    def set_digital_output(self, pin: int, value: bool, **keywords: Any) -> None:
        latch = self.arm.halt_state()
        if latch is not None:
            self.events.append(("output_refused_halted", pin))
            raise ArmHalted(halted_refusal(latch.reason, f"output {pin} was not switched"))
        super().set_digital_output(pin, value, **keywords)


class ConsoleArm(TaskArm):
    """The task tests' ``TaskArm`` (``tests/_task_fakes.py``) with what the console reads of an arm besides its motions.

    * a connection (``connect``/``disconnect``, ``is_connected``);
    * the halt latch, ``SupportsHalt`` with a real ``HaltState``: a halt pressed while a motion runs (``on_motion``)
      says ``in_motion``; with the brake off, as shipped, that motion runs to its end and nothing after it is sent;
    * the controller's fields (``quick_robot_status``, ``get_robot_status``), ``status`` as the test sets it, the
      latch's reason in ``halted``; clearing a protective stop raises: nothing on the console's path may;
    * the planner's state (``planner_state``: ``ready`` as built, ``off`` with ``planner="off"``) and ``start_planner``.

    Every call the console makes of it lands on ``calls``.
    """

    def __init__(self, log: list[Any], *, planner: str = "ready", brakes: bool = False,
                 start_fails: str = "", start_blocks: "threading.Event | None" = None, **keywords: Any) -> None:
        self._latch: HaltState | None = None
        self._moving = False
        self.calls: list[str] = []
        self.status: RobotStatus = RUNNING
        self.status_raises: BaseException | None = None
        self._planner = planner
        self.brakes = brakes
        self.start_fails = start_fails
        self.start_blocks = start_blocks
        super().__init__(log, **keywords)
        self.is_connected = False

    # ---- the connection ---------------------------------------------------------------------------------------

    def connect(self) -> None:
        self.calls.append("connect")
        self.is_connected = True

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self.is_connected = False

    # ---- the halt latch -------------------------------------------------------------------------------------

    @property
    def halted(self) -> str:  # type: ignore[override]
        return "" if self._latch is None else self._latch.reason

    @halted.setter
    def halted(self, value: str) -> None:
        if value:
            self.halt(value)
        else:
            self._latch = None

    def halt(self, reason: str) -> HaltState:
        self.calls.append("halt")
        if self._latch is None:
            self._latch = HaltState(reason=str(reason) or "halt requested", requested_at=time.time(),
                                    in_motion=self._moving)
        return self._latch

    def clear_halt(self) -> None:
        self.calls.append("clear_halt")
        self._latch = None

    def halt_state(self) -> HaltState | None:
        return self._latch

    def brakes_in_motion(self) -> bool:
        return self.brakes

    def _motion(self, kind: str, target: Any, command: Any, **keywords: Any) -> Any:
        self._moving = True
        try:
            result = super()._motion(kind, target, command, **keywords)
        finally:
            self._moving = False
        latch = self._latch
        if latch is not None and latch.in_motion and latch.brake == "pending":
            # The move in flight ran to its end (brakes off): what the UR's moving thread writes once it stands.
            self._latch = dataclasses.replace(latch, brake="ran_out")
        return result

    # ---- the controller -------------------------------------------------------------------------------------

    def _status(self) -> RobotStatus:
        if self.status_raises is not None:
            raise self.status_raises
        if not self.is_connected:
            raise RobotConnectionError("ConsoleArm is not connected")
        return dataclasses.replace(self.status, halted=self.halted)

    def quick_robot_status(self) -> RobotStatus:
        self.calls.append("quick_robot_status")
        return self._status()

    def get_robot_status(self) -> RobotStatus:
        self.calls.append("get_robot_status")
        self.log.append(("status",))
        return self._status()

    def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
        self.calls.append("recover_from_protective_stop")
        raise AssertionError("nothing on the console's path may clear a protective stop")

    # ---- the planner ----------------------------------------------------------------------------------------

    @property
    def planner_state(self) -> str:
        return self._planner

    def start_planner(self) -> dict[str, Any]:
        self.calls.append("start_planner")
        self._planner = "starting"
        if self.start_blocks is not None:
            self.start_blocks.wait(timeout=20)
        if self.start_fails:
            self._planner = "off"
            raise RuntimeError(self.start_fails)
        self._planner = "ready"
        return {"provenance": {"arm": "ur10"}, "arm_descriptor_sha256": "abc"}

    # ---- the carried part, as the console reads it -------------------------------------------------------------

    def detach_payload(self) -> bool:
        self.calls.append("detach_payload")
        return super().detach_payload()


# ---------------------------------------------------------------------------------------------------------------------
# The scripted cell: the service
# ---------------------------------------------------------------------------------------------------------------------


class VirtualClock:
    """A monotonic clock a scripted pick advances by its seconds: a part's ``duration_s`` is the script's, not the
    test machine's. Patched in as ``src.robot.execution.task.time``, which reads ``monotonic`` alone."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


@dataclass
class Scripted:
    """One scripted pick, as the pick loop plays it.

    ``kind``: ``part`` (gripped and lifted), ``push`` (the part boxed in: pushed 30 mm, looked again, then gripped),
    ``empty`` (nothing seen), ``failed`` (a part seen and no grasp), ``kept_out`` (the one part seen stands at ``xy``:
    where a region of the task keeps it out, the pick loop skips it and the pick stops there; anywhere else it is
    gripped like a ``part``). ``looks`` are the looks it perceived from, as the events say them; ``seconds`` how long it
    takes on the :class:`VirtualClock`. ``on_executing`` runs as the grasp begins to move (a test presses a button
    there); ``then`` after the pick returned.
    """

    kind: str = "part"
    xy: tuple[float, float] = PART_XY
    looks: tuple[str, ...] = ("look_1",)
    seconds: float = 0.0
    segmentation_count: int = 3
    candidates: int = 14
    score: float = 0.87
    push_mm: float = 30.0
    on_executing: "Callable[[], None] | None" = None
    then: "Callable[[], None] | None" = None


def _png(seed: int) -> bytes:
    """A small real PNG, different for every seed: what a calculator renders as its overlay."""
    import cv2  # noqa: PLC0415

    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 1] = (seed * 37) % 256
    image[seed % 24, :, 2] = 255
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return bytes(encoded.tobytes())


class ConsoleService(TaskService):
    """The task tests' ``TaskService`` with what the console drives besides a pick: progress events as the pick loop emits them,
    the overlay the calculator renders when the grasp is decided, the service's latch and its acknowledgement, a push
    distance, and the scripted picks played on the real arm and hand."""

    def __init__(self, arm: ConsoleArm, jaws: Any, picks: "Iterable[Scripted | str]" = (), *,
                 clock: "VirtualClock | None" = None, **keywords: Any) -> None:
        super().__init__(arm, jaws, (), **keywords)
        self.scripted = [pick if isinstance(pick, Scripted) else Scripted(pick) for pick in picks]
        self.clock = clock
        self.listener: Any = None
        self.rendered: bytes | None = None
        self.renders = 0
        self.acknowledged = 0

    # ---- what the console attaches -----------------------------------------------------------------------------

    def attach_progress_listener(self, listener: Any) -> None:
        self.listener = listener

    @property
    def last_debug_image_png(self) -> bytes | None:
        return self.rendered

    def acknowledge_needs_person(self) -> None:
        self.acknowledged += 1
        self._needs_person = ""

    def push_distance(self, push_mm: Any = None) -> float:
        if push_mm is None:
            return 30.0
        value = float(push_mm)
        if not 10.0 <= value <= 50.0:
            raise ValueError(f"a push of {value:g} mm is refused: 10 to 50 mm")
        return value

    # ---- a pick -------------------------------------------------------------------------------------------------

    def _emit(self, stage: PickStage, **fields: Any) -> None:
        if self.listener is not None:
            self.listener(PickProgress(stage=stage, **fields))

    def pick(self, **keywords: Any) -> Any:
        self.calls.append(dict(keywords))
        self.zones_seen.append(tuple(self.campaign.zones.regions()))
        if not self.scripted:
            raise AssertionError("the task picked more often than the test scripted")
        pick = self.scripted.pop(0)
        if self._needs_person:
            return _report(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED,
                           telemetry={"stage": "before_start", "stopped_where_the_arm_stands": self._needs_person})
        if self.cancel_check is not None and self.cancel_check():
            self._emit(PickStage.CANCELLED, attempt=0)
            return _report(AutonomousGraspOutcome.CANCELLED, telemetry={"cancelled_before_start": True})
        report = self._play_scripted(pick)
        if self.clock is not None:
            self.clock.advance(pick.seconds)
        if pick.then is not None:
            pick.then()
        return report

    # The events carry the fields, and the words, the pick loop's own carry (``pick_loop.PickLoop``): a pick's outcome
    # as ``str(PickOutcome.X)``, a look's label in ``extra``, the looks' tally on ``ranked``, no attempt count after
    # ``attempt_started``. The console's event logs are captured on these (``scripts/console/capture_event_log.py``).

    def _look(self, attempt: int, pick: Scripted) -> None:
        for look in pick.looks:
            joints = LOOK_1 if look == "look_1" else LOOK_2
            self.arm.move_to_joints(joints)
            count = 0 if pick.kind == "empty" else pick.segmentation_count
            self._emit(PickStage.PERCEIVED, attempt=attempt, segmentation_count=count, extra={"look": look})

    def _finished(self, outcome: PickOutcome) -> None:
        self._emit(PickStage.PICK_FINISHED, outcome=str(outcome))

    def _failed(self, attempt: int, result: Any) -> Any:
        """The attempt a motion the arm refused ended: nothing after it is commanded."""
        self._emit(PickStage.ATTEMPT_FINISHED, attempt=attempt, action="execution_failed",
                   outcome=str(PickOutcome.EXECUTION_FAILED), motion_status=result.status.value,
                   motion_message=result.message)
        self._finished(PickOutcome.EXECUTION_FAILED)
        attempt_row = SimpleNamespace(reasons=(), excluded=None, motion_status=result.status.value,
                                      motion_message=result.message)
        return _report(AutonomousGraspOutcome.EXECUTION_FAILED,
                       pick_report=SimpleNamespace(outcome=PickOutcome.EXECUTION_FAILED, attempts=(attempt_row,)))

    def _play_scripted(self, pick: Scripted) -> Any:
        from src.geometry import Pose  # noqa: PLC0415

        x, y = pick.xy
        self._emit(PickStage.PICK_STARTED, attempt_total=5)
        attempt = 0
        looks = list(pick.looks)
        if pick.kind == "push":
            # Every candidate collided with a neighbour: the part is pushed, the look taken again from where the arm
            # stood, and the next attempt goes on with that look (it perceives nothing new of its own).
            self._emit(PickStage.ATTEMPT_STARTED, attempt=0, attempt_total=5)
            self._look(0, pick)
            beside = Pose.tool_down(x - 40.0, y, GRASP_Z_MM + 10.0, label="push_00")
            self.arm.move(beside)
            self.arm.move(Pose.tool_down(x - 10.0, y, GRASP_Z_MM + 10.0, label="push_01"), linear=True)
            again = f"{pick.looks[-1]} after the push"
            self.arm.move_to_joints(LOOK_1 if pick.looks[-1] == "look_1" else LOOK_2)
            self._emit(PickStage.PERCEIVED, attempt=0, segmentation_count=pick.segmentation_count,
                       extra={"look": again})
            self._emit(PickStage.ATTEMPT_FINISHED, attempt=0, action="push", outcome="pushed", target_index=0,
                       reasons=("all_collided",),
                       extra={"push": "pushed", "push_mm": pick.push_mm,
                              "push_reason": f"Pushed the part {pick.push_mm:.0f} mm and looked again from the look.",
                              "looked_again": again})
            looks.append(again)
            attempt = 1
            self._emit(PickStage.ATTEMPT_STARTED, attempt=attempt, attempt_total=5)
        else:
            self._emit(PickStage.ATTEMPT_STARTED, attempt=attempt, attempt_total=5)
            self._look(attempt, pick)
        if pick.kind == "empty":
            # Nothing segmented at any look: the pick ends there, before anything is ranked.
            self._finished(PickOutcome.NO_PERCEPTION)
            return _report(AutonomousGraspOutcome.NO_TARGET,
                           pick_report=SimpleNamespace(outcome=PickOutcome.NO_PERCEPTION, attempts=()))
        if pick.kind == "kept_out" and self.campaign.zones.kept_out_by_a_region(label="", centre_mm=(x, y, GRASP_Z_MM)):
            # The one part seen stands in a region the task keeps out (the drop it placed it at): the pick loop skips
            # it, and with nothing else seen the pick stops there, saying so (``PickAttempt.excluded_by_regions``).
            self._finished(PickOutcome.RESCANNED_EXHAUSTED)
            row = SimpleNamespace(reasons=(), excluded_by_regions=True,
                                  excluded="Every part the camera sees (1) stands where the task keeps out, so the "
                                           "pick stops here.")
            return _report(AutonomousGraspOutcome.NO_VALID_GRASP,
                           pick_report=SimpleNamespace(outcome=PickOutcome.RESCANNED_EXHAUSTED, attempts=(row,)))
        if pick.kind == "failed":
            self._emit(PickStage.NO_CANDIDATE, attempt=attempt, target_index=0, reasons=("no_candidates_generated",),
                       action="exhausted")
            self._finished(PickOutcome.RESCANNED_EXHAUSTED)
            row = SimpleNamespace(reasons=("no_candidates_generated",), excluded=None)
            return _report(AutonomousGraspOutcome.NO_VALID_GRASP,
                           pick_report=SimpleNamespace(outcome=PickOutcome.RESCANNED_EXHAUSTED, attempts=(row,)))
        # After a push the target's cloud is the look taken again: the earlier views no longer hold the part.
        fused = [looks[-1]] if pick.kind == "push" else list(pick.looks)
        self._emit(PickStage.RANKED, attempt=attempt, target_index=0, candidate_count=pick.candidates,
                   score=pick.score,
                   extra={"looks": looks, "looks_fused": fused, "jaw_faces_seen": [True, False]})
        if self.rendering:
            # Rendered at compute, before anything moves: a new object for every grasp decided.
            self.renders += 1
            self.rendered = _png(self.renders)
        self._emit(PickStage.EXECUTING, attempt=attempt, target_index=0, score=pick.score,
                   position_mm=(x, y, GRASP_Z_MM))
        if pick.on_executing is not None:
            pick.on_executing()
        grasp = Pose.tool_down(x, y, GRASP_Z_MM, label="approach_01")
        above = Pose.tool_down(x, y, GRASP_Z_MM + 80.0, label="approach_00")
        self.arm.detach_payload()
        for target, linear in ((above, False), (grasp, True)):
            moved = self.arm.move(target, linear=linear)
            if moved.status is not MotionStatus.EXECUTED:
                return self._failed(attempt, moved)
        latch = self.arm.halt_state()
        if latch is not None:
            refused = SimpleNamespace(status=MotionStatus.CANCELLED,
                                      message=halted_refusal(latch.reason, "the jaws were not closed"))
            return self._failed(attempt, refused)
        self.jaws.set_closed(True)
        self.arm.attach_payload(40.0)
        lifted = self.arm.move(above, linear=True)
        if lifted.status is not MotionStatus.EXECUTED:
            return self._failed(attempt, lifted)
        self._emit(PickStage.ATTEMPT_FINISHED, attempt=attempt, action="executed", outcome=str(PickOutcome.EXECUTED))
        self._finished(PickOutcome.EXECUTED)
        telemetry: dict[str, Any] = {}
        if pick.kind == "push":
            telemetry["pushes"] = [{"part": 1, "outcome": "pushed", "distance_mm": pick.push_mm}]
        pick_report = SimpleNamespace(outcome=PickOutcome.EXECUTED, attempts=(), gripper_present=True,
                                      object_detected=None, grasp_pose=grasp, object_centre_mm=(x, y, GRASP_Z_MM))
        report = _report(AutonomousGraspOutcome.SUCCEEDED, pick_report=pick_report, telemetry=telemetry)
        return dataclasses.replace(report, looks=tuple(looks), looks_fused=tuple(fused),
                                   jaw_faces_seen=(True, False), hand_eye_gap_mm=2.1)


# ---------------------------------------------------------------------------------------------------------------------
# A camera's target with a picture
# ---------------------------------------------------------------------------------------------------------------------


class SightedLocator(ScriptedLocator):
    """The task tests' ``ScriptedLocator`` that keeps a colour image of every sighting, as a locator the service lends keeps
    (``place_target.locators_for_service`` with ``keep_image``): the target's overlay is drawn over it."""

    def __init__(self, sees: Mapping[str, Any], **keywords: Any) -> None:
        from src.robot.execution import place_target  # noqa: PLC0415

        super().__init__(sees, **keywords)
        self.sightings = place_target._Sightings(self.rig_id, keep=True)  # noqa: SLF001
        place_target._SIGHTINGS[self] = self.sightings  # noqa: SLF001

    def locate(self, prompt: str) -> Any:
        located = super().locate(prompt)
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 40
        self.sightings.show_located(located, image, prompt=prompt)
        return located


# ---------------------------------------------------------------------------------------------------------------------
# The scripted cell, installed in a console
# ---------------------------------------------------------------------------------------------------------------------


def _toggle(log: list[Any], arm: ConsoleArm) -> JawIOGripper:
    """The owner's hand, not yet connected: a Hand-E single_toggle on tool DO0, no sensor, 49.99 / 5.0 mm; asked where
    its jaws stand at connect, a person answers open."""
    return JawIOGripper(_LatchedIO(log, arm), actuation="single_toggle", close_output_pin=0, pulse_s=0.0,
                        close_settle_s=0.0, min_width_mm=5.0, max_width_mm=49.99, ask=lambda _question: "open",
                        sleep=lambda _s: None)


class AnswersOpen:
    """A structured jaws question seam (the library's ``answer_questions_with``) on which a person answers "open" at
    once, where "open" is offered, else "abort" (nothing is sent).

    A console build hands every hand a structured seam (``api.jaws.install``), and a connect refuses a toggle without
    one (``jaws_seam_missing``): the console never asks at the terminal of the server. The scripted cell is installed
    without a build, so it is handed this seam instead of the browser's: the question at its connect is no part of a
    run's story, and publishes nothing on the cell's stream. ``tests/test_api_jaws.py`` asks it in the browser, and
    ``scripts/console/capture_event_log.py`` captures that round trip (``jaws_question_round_trip``)."""

    def __init__(self) -> None:
        self.asked: list[Any] = []

    def __call__(self, asking: Any) -> str:
        self.asked.append(asking)
        return "open" if "open" in tuple(asking.choices) else "abort"


@dataclass
class ScriptedCell:
    """One scripted cell: the shared log, the arm, the hand and the service, and the clock its picks advance."""

    log: list[Any]
    arm: ConsoleArm
    jaws: Any
    service: ConsoleService
    clock: VirtualClock = field(default_factory=VirtualClock)

    @classmethod
    def build(cls, picks: "Iterable[Scripted | str]" = (), *, hand: Any = None, wrist: bool = False,
              looks: tuple[Any, ...] = (), arm_keywords: "Mapping[str, Any] | None" = None,
              **service_keywords: Any) -> "ScriptedCell":
        log: list[Any] = []
        arm = ConsoleArm(log, **dict(arm_keywords or {}))
        jaws = hand if hand is not None else _toggle(log, arm)
        clock = VirtualClock()
        service = ConsoleService(arm, jaws, picks, clock=clock, wrist=wrist, looks=looks, **service_keywords)
        return cls(log=log, arm=arm, jaws=jaws, service=service, clock=clock)

    def install(self, console: Any, *, connect: bool = True) -> None:
        """Adopt the service as the console's built cell and, with ``connect``, bring it up through the session's own
        connect (a preview token, the arm, then the hand, which asks where its jaws stand).

        The hand is handed a structured question seam first, as a build hands one (:class:`AnswersOpen`, with the
        console's own gate before a change of the output), so every connect of it, this one or a later one through
        ``POST /v1/cell/connect``, is answered "open" at once."""
        console.session.adopt(self.service)
        installer = getattr(self.jaws, "answer_questions_with", None)
        if callable(installer):
            from api import jaws as browser_jaws  # noqa: PLC0415

            installer(AnswersOpen(), before_change=browser_jaws.before_change(console))
        if connect:
            preview = console.session.preview(console.robot(), console.fingerprint())
            console.session.connect(preview.token, console.fingerprint(), None)
        del self.log[:]

    def do0_changes(self) -> int:
        from tests._task_fakes import do0_changes  # noqa: PLC0415

        return do0_changes(self.log)

    def motions(self) -> list[Any]:
        from tests._task_fakes import motions  # noqa: PLC0415

        return motions(self.log)

    @property
    def payload(self) -> PayloadModel:
        return self.arm.payload_model()


# ---------------------------------------------------------------------------------------------------------------------
# A console on a scratch tree, installed as the app's
# ---------------------------------------------------------------------------------------------------------------------


class ConsoleCase(unittest.TestCase):
    """A fresh console on a scratch copy of the shipped tree (:func:`dummy_tree`), installed as the process console for
    the app, with a client. The countdown ticks fast (``countdown_step_s``) so a test of it takes a fraction of a
    second; its three steps and their order are the real ones."""

    gripper = "dummy"
    poses = True
    default_place = True

    def setUp(self) -> None:
        from fastapi.testclient import TestClient  # noqa: PLC0415

        from api.app import create_app  # noqa: PLC0415
        from api.cell import Console, set_console  # noqa: PLC0415

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        profile = dummy_tree(self.tmp, gripper=self.gripper, poses=self.poses, default_place=self.default_place)
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=profile)
        self.cell.record_log_path = self.tmp.parent / "grasp_records.jsonl"
        self.cell.registry.countdown_step_s = 0.05
        self._previous_console = set_console(self.cell)
        self.client = TestClient(create_app())
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from api.cell import set_console  # noqa: PLC0415

        try:
            self.cell.session.release()
        except Exception:  # pragma: no cover - cleanup must not mask a failure
            pass
        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    # ---- bringing a cell up ---------------------------------------------------------------------------------------

    def build_dummy(self, *, connect: bool = True) -> None:
        """console_dummy: the rehearsal build, then the connect through the preview token."""
        built = self.client.post("/v1/cell/build", params={"rehearse": True})
        self.assertEqual(200, built.status_code, built.text)
        if connect:
            token = self.client.get("/v1/cell/connect-preview").json()["token"]
            up = self.client.post("/v1/cell/connect", json={"token": token})
            self.assertEqual(200, up.status_code, up.text)

    def scripted(self, picks: "Iterable[Scripted | str]" = (), *, connect: bool = True,
                 **keywords: Any) -> ScriptedCell:
        """A scripted cell, built and (with ``connect``) connected in this console."""
        cell = ScriptedCell.build(picks, **keywords)
        cell.install(self.cell, connect=connect)
        return cell

    # ---- runs ------------------------------------------------------------------------------------------------------

    def task(self, *, expect: int = 202, **body: Any) -> Any:
        """POST /v1/task with ``body`` (a pose place at the default unless it says otherwise)."""
        body.setdefault("place", {"kind": "pose", "pose": None})
        answered = self.client.post("/v1/task", json=body)
        self.assertEqual(expect, answered.status_code, answered.text)
        return answered.json()

    def finished(self, run_id: str, *, timeout: float = 30.0) -> Any:
        """The run once ended, as ``GET /v1/runs/{id}`` answers it."""
        await_run(self.cell, run_id, timeout=timeout)
        body = self.client.get(f"/v1/runs/{run_id}").json()
        self.assertNotEqual("running", body["state"], body)
        return body

    def refused(self, answered: Any, status: int, code: str) -> dict[str, Any]:
        self.assertEqual((status, code), (answered.status_code, answered.json().get("code")), answered.text)
        return answered.json()
