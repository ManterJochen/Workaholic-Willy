"""Doubles for the task tests: the owner's cell in miniature, every motion and every change of DO0 on one log.

* :class:`TaskArm` is a UR-like arm whose motions go through a planner (``line_motion`` CHECKED, ``plans_paths``), so
  ``Robot.place``, ``Robot.home`` and ``Robot.move_joints`` run on it as on the owner's cuRobo UR. It writes every
  motion on the shared log, answers ``fk``, ``nearest_configuration`` and ``screen_configuration`` from tables a test
  scripts, models a carried part (``CarriesPayload``), reports its controller (``SupportsRobotStatus``; clearing a stop
  raises: nothing on the task's path may), and holds a live planner world that holds a pick's frames.
* The hand is the real ``JawIOGripper`` single_toggle on tool DO0 over the recording I/O of
  ``tests/test_a_stopped_controller_moves_no_jaws.py``: a change of DO0 is read off the outputs, never off a double's
  bookkeeping (:func:`do0_changes`).
* :class:`TaskService` is the pick service as a task drives it, duck-typed: each ``pick()`` plays the next scripted
  pick on the real arm and hand (a part picked closes the toggle once, with the arm's motions around it), and
  ``put_back`` is the service's own (``AutonomousGraspService.put_back``).
* :class:`RecordingHooks` records every event of the task and answers the operator's buttons as a test sets them.
* :class:`ScriptedLocator` stands in for a ``Locator`` over the service's camera: it answers each prompt with what a
  test says the camera sees from where the arm stands.
* :class:`MeasuringHand` is a hand that is not a toggle: it closes and opens by intent and measures whether it holds a
  part, so a part that sticks after the jaws opened is still measured.

A bin is :func:`bin_points`: its four walls' tops at the rim, their inner faces and its floor, as a camera above sees
them, sampled every few millimetres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from src.contracts import UNSET, chosen
from src.geometry import Pose
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotCapabilities,
    RobotMode,
    RobotStatus,
    SafetyMode,
)
from src.robot.core.arm_capabilities import LineMotion, LineReading, PayloadModel
from src.robot.core.errors import RobotError, RobotKinematicsError
from src.robot.core.gripper import HoldEvidence
from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
from src.robot.execution.autonomous_grasp.prompt import PickPrompt
from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome, AutonomousGraspReport
from src.robot.grasping.loop.pick_loop import PickOutcome
from src.robot.grasping.recovery.exclusion_zones import ExclusionZones
from src.robot.grippers.jaw_io import JawIOGripper
from src.robot.perception.locator import Located, LocatedObject
from src.robot.safety.planning.band import PoseScreen, PoseVerdict
from tests.test_a_stopped_controller_moves_no_jaws import _IO, _Write

__all__ = [
    "BIN_CENTRE",
    "BIN_RIM_MM",
    "BIN_SIZE",
    "GRASP_Z_MM",
    "HOME",
    "MeasuringHand",
    "PART_XY",
    "PLACE_JOINTS",
    "PLACE_TCP",
    "PARK_JOINTS",
    "PARK_TCP",
    "RUNNING",
    "PROTECTIVE",
    "RecordingHooks",
    "ScriptedLocator",
    "TaskArm",
    "TaskService",
    "bin_object",
    "bin_points",
    "do0_changes",
    "POSES",
    "Pick",
    "Ran",
    "located",
    "motions",
    "run",
    "toggle",
]

RUNNING = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                      emergency_stopped=False)
PROTECTIVE = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP,
                         protective_stopped=True, emergency_stopped=False, message="Safetystatus: PROTECTIVE_STOP")
_CAPS = RobotCapabilities(
    vendor="ur", model="ur10", dof=6, supports_joint_move=True, supports_linear_move=True,
    supports_async_move=False, has_native_fk=True, has_native_ik=True, has_force_control=False, is_simulated=False,
)

#: Where the arm's home puts the tool, and the arm's home joints.
HOME = Pose.tool_down(0.0, -500.0, 400.0, label="home")
HOME_JOINTS = JointPositions.deg(0.0, -90.0, -90.0, -90.0, 90.0, 0.0)
#: A taught place pose: where the fingertips stood when it was taught, the part's bottom let go there.
PLACE_JOINTS = JointPositions.deg(-60.0, -95.0, -120.0, -55.0, 90.0, 0.0)
PLACE_TCP = Pose.tool_down(300.0, -400.0, 150.0, label="drop_left")
#: A taught park pose, a return the operator chose instead of home.
PARK_JOINTS = JointPositions.deg(30.0, -80.0, -100.0, -90.0, 90.0, 0.0)
PARK_TCP = Pose.tool_down(-200.0, -450.0, 380.0, label="park")
#: Where the scripted parts stand on the bench (BASE XY), and the Z the tool closes at on one: 40 mm parts on a bench
#: at 0, grasped 20 mm above it.
PART_XY = (150.0, -650.0)
GRASP_Z_MM = 20.0
#: A bin on the bench beside the base: its centre, its outer size and its rim.
BIN_CENTRE = (400.0, -300.0)
BIN_SIZE = (300.0, 200.0)
BIN_RIM_MM = 120.0


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(value, 1) for value in joints.degrees())


def motions(log: Iterable[Any]) -> list[tuple[Any, ...]]:
    """The arm's motions on ``log``, in order: ``("move", xyz, linear)``, ``("joints", key)``, ``("home",)``."""
    return [entry for entry in log if isinstance(entry, tuple) and entry and entry[0] in ("move", "joints", "home")]


def do0_changes(log: Iterable[Any]) -> int:
    """How often DO0 changed on ``log``: each change moves a toggle hand once, switched on or off."""
    return sum(1 for entry in log if isinstance(entry, _Write) and entry[1] == 0 and entry.changed)


class _ArmIO(_IO):
    """The tool I/O as the arm drives it: a write while the arm's halt latch is set raises, nothing sent (L9)."""

    def __init__(self, log: list[Any], arm: Any) -> None:
        super().__init__(log)
        self.arm = arm

    def set_digital_output(self, pin: int, value: bool, **keywords: Any) -> None:
        if getattr(self.arm, "halted", ""):
            self.events.append(("output_refused_halted", pin))
            raise RobotError(f"the arm is halted ({self.arm.halted}): output {pin} was not switched")
        super().set_digital_output(pin, value, **keywords)


def toggle(log: list[Any], *, closed: bool = False, arm: Any = None) -> JawIOGripper:
    """The owner's hand: a Hand-E single_toggle on tool DO0, no sensor, 49.99 / 5.0 mm, a person who says open.

    Connected, its count OPEN (CLOSED where ``closed``), and the connect's writes taken off ``log``. With ``arm``, its
    I/O is the arm's: a write while the arm is halted raises with nothing sent, as on the UR.
    """
    io = _IO(log) if arm is None else _ArmIO(log, arm)
    jaws = JawIOGripper(io, actuation="single_toggle", close_output_pin=0, pulse_s=0.0, close_settle_s=0.0,
                        min_width_mm=5.0, max_width_mm=49.99, ask=lambda _question: "open", sleep=lambda _s: None)
    jaws.connect()
    if closed:
        jaws.set_closed(True)
    del log[:]
    return jaws


class _World:
    """The arm's live planner world as the task meets it: it holds a pick's frames, and keeps offers out."""

    def __init__(self, log: list[Any], *, holds: bool = True) -> None:
        self.log = log
        self.holds = holds
        self.holding = False
        self.offers = 0

    def hold_pick_views(self) -> bool:
        self.log.append(("hold",))
        self.holding = True
        return self.holds

    def forget_pick_views(self) -> None:
        self.log.append(("forget",))
        self.holding = False

    @property
    def holds_pick_views(self) -> bool:
        return self.holding

    def offer_segmentation(self, **_: Any) -> None:
        self.offers += 1
        self.log.append(("offer",))

    def forget_segmentation(self) -> None:
        self.log.append(("forget_offer",))


class TaskArm:
    """A UR-like arm whose motions a planner plans and judges, writing every motion on ``log``.

    ``fk_table`` maps joints (rounded degrees) to the TCP pose they put the tool at; ``screens`` maps joints to the
    verdict ``screen_configuration`` gives (CLEAR otherwise); ``unreachable_above_mm`` makes ``nearest_configuration``
    refuse every pose whose Z stands above it. ``refuse`` decides, per motion, whether the planner refuses it before
    anything is sent (a status) or nothing: ``refuse(kind, index, target) -> MotionStatus | None``. ``on_motion`` runs
    before each motion runs, so a test can press a button while the arm is on its way (it may raise, as a driver that
    meets a camera which cannot vouch does). ``statuses`` are what the controller answers, in order, the last one from
    then on. ``attaches`` is what an attach of the carried part leaves in force: the planner and the filter as on a
    cuRobo UR that models it, ``FILTER_ONLY`` where the planner declined it, ``NONE`` where nothing modelled it.
    """

    plans_paths = True

    def __init__(self, log: list[Any], *, fk_table: "Mapping[tuple[float, ...], Pose] | None" = None,
                 screens: "Mapping[tuple[float, ...], PoseVerdict] | None" = None,
                 unreachable_above_mm: float | None = None,
                 refuse: "Callable[[str, int, Any], MotionStatus | None] | None" = None,
                 on_motion: "Callable[[str, int], None] | None" = None,
                 statuses: tuple[RobotStatus, ...] = (RUNNING,), declined: str | None = None,
                 world: bool = True, holds: bool = True,
                 attaches: PayloadModel = PayloadModel.PLANNER_AND_FILTER) -> None:
        self.log = log
        self.fk_table = {**{_key(PLACE_JOINTS): PLACE_TCP, _key(PARK_JOINTS): PARK_TCP, _key(HOME_JOINTS): HOME},
                         **dict(fk_table or {})}
        self.screens = dict(screens or {})
        self.unreachable_above_mm = unreachable_above_mm
        self.refuse = refuse
        self.on_motion = on_motion
        self._statuses = list(statuses)
        self.declined = declined
        self.attaches = attaches
        self.live_planner_world: Any = _World(log, holds=holds) if world else None
        self.is_connected = True
        self.home_joint_positions = tuple(HOME_JOINTS.tolist())
        self._tcp = HOME
        self._joints = HOME_JOINTS
        self._payload: float | None = None
        self.index = 0
        self.screened: list[tuple[float, ...]] = []
        self.nearest_asked: list[Pose] = []
        self.moved: list[Pose] = []
        self.halted = ""

    # ---- what the arm says --------------------------------------------------------------------------------

    @property
    def capabilities(self) -> RobotCapabilities:
        return _CAPS

    def line_motion(self) -> LineReading:
        return LineReading(LineMotion.CHECKED, "every sample is judged by the exact mesh guard and the planner")

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def get_joint_positions(self) -> JointPositions:
        return self._joints

    def get_robot_status(self) -> RobotStatus:
        self.log.append(("status",))
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
        raise AssertionError("nothing on a task's path may clear a protective stop")

    def halt_state(self) -> Any:
        return SimpleNamespace(reason=self.halted, requested_at=0.0) if self.halted else None

    def wait_until_steady(self, timeout_s: float = 5.0, poll_interval_s: float = 0.02) -> bool:
        return True

    # ---- kinematics and screening -------------------------------------------------------------------------

    def fk(self, joints: JointPositions) -> Pose:
        found = self.fk_table.get(_key(joints))
        if found is None:
            raise RobotKinematicsError(f"no FK for {joints.degrees()}")
        return found

    def nearest_configuration(self, pose: Pose) -> JointPositions:
        self.nearest_asked.append(pose)
        if self.unreachable_above_mm is not None and float(pose.position_mm[2]) > self.unreachable_above_mm:
            raise RobotKinematicsError(f"{pose.label or 'pose'}: refused by the endpoint gate: out of reach")
        x, y, z = (float(v) for v in pose.position_mm)
        return JointPositions.deg(round(x / 10.0, 1), round(y / 10.0, 1), round(z / 10.0, 1), -90.0, 90.0, 0.0)

    def screen_configuration(self, joints: JointPositions, *, ask_planner: bool = True) -> PoseScreen:
        key = _key(joints)
        self.screened.append(key)
        verdict = self.screens.get(key, PoseVerdict.CLEAR)
        nearby = tuple(v + 0.01 for v in joints.tolist()) if verdict is not PoseVerdict.CLEAR else None
        return PoseScreen(verdict, f"scripted {verdict.value}", nearby=nearby)

    # ---- the carried part ---------------------------------------------------------------------------------

    def attach_payload(self, grip_width_mm: float) -> bool:
        self.log.append(("attach",))
        if self.attaches is PayloadModel.NONE:
            return False
        self._payload = float(grip_width_mm)
        return self.attaches is PayloadModel.PLANNER_AND_FILTER

    def detach_payload(self) -> bool:
        self.log.append(("detach",))
        self._payload = None
        return True

    def payload_declined_reason(self) -> str | None:
        return self.declined

    def payload_model(self) -> PayloadModel:
        return PayloadModel.NONE if self._payload is None else self.attaches

    # ---- motions ------------------------------------------------------------------------------------------

    def _motion(self, kind: str, target: Any, command: MotionCommand, *, linear: bool = False,
                pose: Pose | None = None, joints: JointPositions | None = None) -> MotionResult:
        index = self.index
        self.index += 1
        if self.halted:
            # A latched arm refuses the next motion before anything is sent (L9).
            self.log.append(("refused_halted", kind))
            return MotionResult.failed(MotionStatus.CANCELLED, command, target_pose=pose, target_joints=joints,
                                       message=f"halted: {self.halted}; nothing was sent")
        if self.on_motion is not None:
            # A button pressed while this motion runs: with robot.ur.brake_on_halt off (Q1), the motion runs to its end.
            self.on_motion(kind, index)
        refused = self.refuse(kind, index, target) if self.refuse is not None else None
        if refused is not None:
            self.log.append(("refused", kind, refused.value))
            return MotionResult.failed(refused, command, target_pose=pose, target_joints=joints,
                                       message=f"scripted refusal of {kind} {index}")
        if kind == "move":
            assert pose is not None
            self.log.append(("move", tuple(round(float(v), 1) for v in pose.position_mm), linear))
            self.moved.append(pose)
            self._tcp = pose
        elif kind == "joints":
            assert joints is not None
            self.log.append(("joints", _key(joints)))
            self._joints = joints
            self._tcp = self.fk_table.get(_key(joints), self._tcp)
        else:
            self.log.append(("home",))
            self._joints = HOME_JOINTS
            self._tcp = HOME
        return MotionResult.executed(command, target_pose=pose, target_joints=joints, message=kind)

    def move(self, pose: Pose, *, linear: bool = False, **_: Any) -> MotionResult:
        return self._motion("move", pose, MotionCommand.MOVE_TO, linear=linear, pose=pose)

    def move_to_joints(self, joints: JointPositions, **_: Any) -> MotionResult:
        return self._motion("joints", joints, MotionCommand.MOVE_JOINTS, joints=joints)

    def move_to_home(self) -> MotionResult:
        return self._motion("home", None, MotionCommand.MOVE_HOME, joints=HOME_JOINTS)


# ---------------------------------------------------------------------------------------------------------------------
# The pick service, scripted
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Pick:
    """One scripted pick: what it comes to (``kind``), where the part stood, and what else it says.

    ``kind`` is ``part`` (gripped and lifted), ``empty`` (nothing found), ``excluded`` (only parts the task keeps out
    were seen, each in a region of the task), ``zoned`` (only parts were seen that a ``next_target`` zone skips after a
    failed pick: parts the task still has to pick), ``failed`` (a part seen and no grasp), ``failed_holding`` (the lift
    was refused with the part closed in the jaws), ``fault``, ``controller``, ``gripper``, ``needs_person`` or
    ``cancelled``. ``detector_failed`` raises the service's detector failure count during the pick. ``grasp`` False
    reports no grasp pose on a part picked.
    """

    kind: str = "part"
    xy: tuple[float, float] = PART_XY
    grasp: bool = True
    detector_failed: bool = False
    cloud: "np.ndarray | None" = None
    then: "Callable[[], None] | None" = None


def _report(outcome: AutonomousGraspOutcome, *, telemetry: "Mapping[str, Any] | None" = None,
            pick_report: Any = None, fault: Exception | None = None) -> AutonomousGraspReport:
    return AutonomousGraspReport(outcome=outcome, mode=GraspMode.EASY, profile=_profile_for(GraspMode.EASY),
                                 pick_report=pick_report, telemetry=dict(telemetry or {}), fault=fault)


class TaskService:
    """The pick service a task drives, duck-typed, on the real arm and hand: every pick plays the next of ``picks``.

    It keeps what the task set (the prompt, the closing axis, the overlay switch, the cancel check, the campaign and its
    push distance) so a test reads that each was put back. ``start_campaign`` acknowledges the latch a recovery left,
    as the real service's does, so a task that called it before reading the latch would clear it.
    """

    def __init__(self, arm: TaskArm, jaws: Any, picks: "Iterable[Pick | str]" = (), *, wrist: bool = False,
                 looks: tuple[Any, ...] = (), grounds: bool = True, needs_person: str = "",
                 natural_axis: Any = None, failure_log: "list[Any] | None" = None) -> None:
        self.arm = arm
        self.jaws = jaws
        self.log = arm.log
        self.picks = [pick if isinstance(pick, Pick) else Pick(pick) for pick in picks]
        self.perceives_from_the_wrist = wrist
        self.configured_looks = looks
        self.grounds = grounds
        self._needs_person = needs_person
        self._campaign: Any = None
        self.campaigns: list[dict[str, Any]] = []
        self.prompt = PickPrompt(phrase="object" if grounds else "")
        self.prompts: list[Any] = []
        self.closing_axes: list[Any] = []
        self.closing_axis: Any = None
        self.rendering = False
        self.cancel_check: Any = None
        self.cancel_checks: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self.zones_seen: list[tuple[Any, ...]] = []
        self.rendering_seen: list[bool] = []
        self.cancel_seen: list[Any] = []
        self.failures = 0
        self.failure_log = failure_log if failure_log is not None else []
        self.looked_around: Any = None
        policy = SimpleNamespace(standoff_mm=80.0, closing_axis=None, align_closing_to_base_x=False)
        self.runtime = SimpleNamespace(orchestrator=SimpleNamespace(
            arm=arm, gripper=jaws, policy=policy, natural_closing_axis=natural_axis))

    # ---- what the task reads and sets ---------------------------------------------------------------------

    @property
    def stopped_where_the_arm_stands(self) -> str:
        return self._needs_person

    def start_campaign(self, *, push_mm: Any = UNSET, critical_parts: Any = UNSET,
                       recovery_actions: Any = UNSET, blocker_is_the_pick: Any = UNSET) -> Any:
        self.campaigns.append({"push_mm": push_mm if chosen(push_mm) else None,
                               **({"critical_parts": critical_parts} if chosen(critical_parts) else {}),
                               **({"recovery_actions": recovery_actions} if chosen(recovery_actions) else {}),
                               **({"blocker_is_the_pick": blocker_is_the_pick} if chosen(blocker_is_the_pick) else {})})
        self._campaign = SimpleNamespace(zones=ExclusionZones(), distance_mm=push_mm if chosen(push_mm) else 30.0)
        self._needs_person = ""
        self.log.append(("campaign",))
        return self._campaign

    @property
    def campaign(self) -> Any:
        if self._campaign is None:
            self._campaign = SimpleNamespace(zones=ExclusionZones(), distance_mm=30.0)
        return self._campaign

    def set_prompt(self, prompt: Any) -> PickPrompt:
        wanted = PickPrompt.from_text(prompt) if isinstance(prompt, str) else prompt
        previous, self.prompt = self.prompt, wanted
        self.prompts.append(wanted)
        return previous

    def set_closing_axis(self, axis: Any) -> Any:
        from src.geometry.closing_axis import closing_axis_of  # noqa: PLC0415

        wanted = None if axis is None else closing_axis_of(axis)
        previous, self.closing_axis = self.closing_axis, wanted
        self.closing_axes.append(wanted)
        return previous

    def closing_axis_refusal(self, axis: Any) -> str:
        """The service's own read, on this service's policy."""
        from src.robot.execution.autonomous_grasp.service import AutonomousGraspService  # noqa: PLC0415

        return AutonomousGraspService.closing_axis_refusal(self, axis)  # type: ignore[arg-type]

    def _closing_axis_wanted(self, axis: Any) -> Any:
        from src.robot.execution.autonomous_grasp.service import AutonomousGraspService  # noqa: PLC0415

        return AutonomousGraspService._closing_axis_wanted(self, axis)  # type: ignore[arg-type]  # noqa: SLF001

    @property
    def debug_image_rendering_enabled(self) -> bool:
        return self.rendering

    def enable_debug_image_rendering(self, enabled: bool = True) -> None:
        self.rendering = bool(enabled)

    def set_cancel_check(self, check: Any) -> None:
        self.cancel_check = check
        self.cancel_checks.append(check)

    def controller_refusal(self) -> str:
        from src.robot.execution.handling import _controller_refusal  # noqa: PLC0415

        return _controller_refusal(self.arm)

    def detector_failures(self) -> int:
        return self.failures + len(self.failure_log)

    def grounds_a_phrase(self) -> bool:
        return self.grounds

    def put_back(self, report: Any) -> Any:
        from src.robot.execution.autonomous_grasp.service import AutonomousGraspService  # noqa: PLC0415

        return AutonomousGraspService.put_back(self, report)  # type: ignore[arg-type]

    # ---- a pick ---------------------------------------------------------------------------------------------

    def pick(self, **keywords: Any) -> AutonomousGraspReport:
        self.calls.append(dict(keywords))
        self.zones_seen.append(tuple(self.campaign.zones.regions()))
        self.rendering_seen.append(self.rendering)
        self.cancel_seen.append(self.cancel_check)
        if not self.picks:
            raise AssertionError("the task picked more often than the test scripted")
        pick = self.picks.pop(0)
        if pick.detector_failed:
            self.failures += 1
        if self.cancel_check is not None and self.cancel_check():
            return _report(AutonomousGraspOutcome.CANCELLED, telemetry={"cancelled_before_start": True})
        report = self._play(pick)
        if pick.then is not None:
            pick.then()
        return report

    def _play(self, pick: Pick) -> AutonomousGraspReport:
        x, y = pick.xy
        look = self.configured_looks[0] if self.configured_looks else None
        if look is not None and not isinstance(look, str):
            self.arm.move_to_joints(look)
        if pick.kind == "empty":
            return _report(AutonomousGraspOutcome.NO_TARGET,
                           pick_report=SimpleNamespace(outcome=PickOutcome.NO_PERCEPTION, attempts=()))
        if pick.kind == "excluded":
            attempt = SimpleNamespace(reasons=(), excluded="Every part the camera sees (1) stands where the task keeps "
                                                         "out, so the pick stops here.", excluded_by_regions=True)
            return _report(AutonomousGraspOutcome.NO_VALID_GRASP,
                           pick_report=SimpleNamespace(outcome=PickOutcome.RESCANNED_EXHAUSTED, attempts=(attempt,)))
        if pick.kind == "zoned":
            attempt = SimpleNamespace(reasons=(), excluded="Every 'part' the camera sees (1) failed in one of the last 3 "
                                                         "picks and is being skipped, so the pick stops here.",
                                      excluded_by_regions=False)
            return _report(AutonomousGraspOutcome.NO_VALID_GRASP,
                           pick_report=SimpleNamespace(outcome=PickOutcome.RESCANNED_EXHAUSTED, attempts=(attempt,)))
        if pick.kind == "failed":
            attempt = SimpleNamespace(reasons=("no_candidates_generated",), excluded=None)
            return _report(AutonomousGraspOutcome.NO_VALID_GRASP,
                           pick_report=SimpleNamespace(outcome=PickOutcome.RESCANNED_EXHAUSTED, attempts=(attempt,)))
        if pick.kind == "fault":
            return _report(AutonomousGraspOutcome.EXECUTION_FAILED, fault=RuntimeError("the camera stopped delivering"))
        if pick.kind == "controller":
            return _report(AutonomousGraspOutcome.CANCELLED, telemetry={
                "low_level_outcome": str(PickOutcome.CONTROLLER_NOT_OPERATIONAL), "controller": "protective stop"})
        if pick.kind == "gripper":
            return _report(AutonomousGraspOutcome.EXECUTION_FAILED, telemetry={
                "low_level_outcome": str(PickOutcome.GRIPPER_FAULT), "gripper_fault": "the gripper raised"})
        if pick.kind == "needs_person":
            self._needs_person = "a push stopped where the arm stands"
            return _report(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED,
                           telemetry={"push_stopped": "a push stopped where the arm stands"})
        if pick.kind == "cancelled":
            return _report(AutonomousGraspOutcome.CANCELLED, telemetry={"cancelled_before_start": True})
        # A part: the approach, the close (one change of DO0 on a toggle), the lift.
        grasp = Pose.tool_down(x, y, GRASP_Z_MM, label="approach_01")
        above = Pose.tool_down(x, y, GRASP_Z_MM + 80.0, label="approach_00")
        self.arm.detach_payload()
        self.arm.move(above)
        self.arm.move(grasp, linear=True)
        self.jaws.set_closed(True)
        self.arm.attach_payload(40.0)
        if pick.kind == "failed_holding":
            attempt = SimpleNamespace(reasons=(), excluded=None)
            return _report(AutonomousGraspOutcome.EXECUTION_FAILED,
                           pick_report=SimpleNamespace(outcome=PickOutcome.EXECUTION_FAILED, attempts=(attempt,)))
        self.arm.move(above, linear=True)
        if pick.cloud is not None:
            self.looked_around = SimpleNamespace(judged=SimpleNamespace(target_cloud_base_mm=pick.cloud), views=())
        pick_report = SimpleNamespace(outcome=PickOutcome.EXECUTED, attempts=(), gripper_present=True,
                                      object_detected=None, grasp_pose=grasp if pick.grasp else None,
                                      object_centre_mm=(x, y, GRASP_Z_MM))
        return _report(AutonomousGraspOutcome.SUCCEEDED, pick_report=pick_report)


# ---------------------------------------------------------------------------------------------------------------------
# The operator's buttons and the task's events
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class RecordingHooks:
    """What the console's hooks are to a task: every event recorded, and the operator's buttons as the test sets them.

    ``on_event`` runs after an event is recorded, so a test presses a button when the task says a step began.
    """

    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    picks: list[tuple[int, int, Any]] = field(default_factory=list)
    stop: bool = False
    halt: bool = False
    gone: str = ""
    on_event: "Callable[[str, dict[str, Any]], None] | None" = None

    def event(self, name: Any, /, **data: Any) -> None:
        said = data.pop("said", None)
        assert isinstance(said, str) and said, f"{name} carries no sentence"
        self.said.append(said)
        self.events.append((str(name), data))
        if self.on_event is not None:
            self.on_event(str(name), data)

    def pick_done(self, part: int, pick: int, report: Any) -> None:
        self.picks.append((part, pick, report))

    def stop_after_part(self) -> bool:
        return self.stop

    def halted(self) -> bool:
        return self.halt

    def abandoned(self) -> str:
        return self.gone

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def of(self, name: str) -> list[dict[str, Any]]:
        return [data for event, data in self.events if event == name]


class MeasuringHand:
    """A hand that is not a toggle: it closes and opens by intent (``set_closed``) and measures whether it holds a part.

    ``holding`` says it holds one now (a part left in it by a run that stopped); a close holds whatever it closed on,
    and an open lets it go unless ``sticks`` (the part stays in the jaws, still measured). ``raise_on_open`` makes an
    open raise, as a gripper whose link dropped does. Every command is written on ``commands``.
    """

    is_connected = True
    min_width_mm, max_width_mm = 0.0, 50.0

    def __init__(self, *, holding: bool = False, sticks: bool = False, raise_on_open: bool = False) -> None:
        self.closed = holding
        self.holding = holding
        self.sticks = sticks
        self.raise_on_open = raise_on_open
        self.commands: list[bool] = []

    def set_closed(self, closed: bool) -> None:
        if not closed and self.raise_on_open:
            raise RobotError("the gripper's link dropped while it opened")
        self.commands.append(bool(closed))
        self.closed = bool(closed)
        self.holding = True if closed else self.sticks and self.holding

    def get_width_mm(self) -> float:
        return 10.0 if self.closed else self.max_width_mm

    def hold_evidence(self) -> HoldEvidence:
        if self.holding:
            return HoldEvidence.HELD
        return HoldEvidence.EMPTY if self.closed else HoldEvidence.UNMEASURED

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def activate(self) -> None:
        return None

    def set_width_mm(self, width_mm: float, **_: Any) -> None:
        self.set_closed(width_mm < 25.0)

    def is_object_detected(self) -> bool:
        return self.holding


# ---------------------------------------------------------------------------------------------------------------------
# A camera place's target: a bin, located
# ---------------------------------------------------------------------------------------------------------------------


def bin_points(centre_xy: tuple[float, float] = BIN_CENTRE, size_xy: tuple[float, float] = BIN_SIZE,
               rim_mm: float = BIN_RIM_MM, *, yaw_deg: float = 0.0, wall_mm: float = 8.0, floor_mm: float = 5.0,
               step_mm: float = 4.0, inside: bool = True) -> np.ndarray:
    """What a camera above sees of an open bin, BASE mm: its walls' tops at ``rim_mm``, their inner faces down to the
    floor and the floor inside, sampled every ``step_mm``; ``inside`` False is a closed box, its top all at the rim."""
    half_x, half_y = size_xy[0] / 2.0, size_xy[1] / 2.0
    xs = np.arange(-half_x, half_x + 1e-9, step_mm)
    ys = np.arange(-half_y, half_y + 1e-9, step_mm)
    gx, gy = np.meshgrid(xs, ys)
    u, v = gx.ravel(), gy.ravel()
    on_wall = (np.abs(u) >= half_x - wall_mm) | (np.abs(v) >= half_y - wall_mm)
    if inside:
        top = np.column_stack([u[on_wall], v[on_wall], np.full(int(on_wall.sum()), rim_mm)])
        floor = np.column_stack([u[~on_wall], v[~on_wall], np.full(int((~on_wall).sum()), floor_mm)])
        faces = []
        for z in np.arange(floor_mm + step_mm, rim_mm, step_mm):
            ring = (np.isclose(np.abs(u), half_x - wall_mm, atol=step_mm / 2.0) & (np.abs(v) <= half_y - wall_mm)) | (
                np.isclose(np.abs(v), half_y - wall_mm, atol=step_mm / 2.0) & (np.abs(u) <= half_x - wall_mm))
            faces.append(np.column_stack([u[ring], v[ring], np.full(int(ring.sum()), z)]))
        local = np.vstack([top, floor, *faces])
    else:
        local = np.column_stack([u, v, np.full(u.shape[0], rim_mm)])
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    x = centre_xy[0] + c * local[:, 0] - s * local[:, 1]
    y = centre_xy[1] + s * local[:, 0] + c * local[:, 1]
    return np.column_stack([x, y, local[:, 2]])


def bin_object(points: "np.ndarray | None" = None, *, label: str = "blue bin", score: float | None = 0.8) -> LocatedObject:
    """A located object from ``points`` (a bin where none are given), as ``Locator`` places one."""
    cloud = bin_points() if points is None else np.asarray(points, dtype=np.float64)
    centre = None if cloud.shape[0] == 0 else tuple(float(v) for v in np.median(cloud, axis=0))
    return LocatedObject(label=label, score=score, box_px=None, mask=np.ones((4, 4), dtype=bool),
                         points_base_mm=cloud, centre_mm=centre)  # type: ignore[arg-type]


def located(*objects: LocatedObject, camera: str = "wrist", at: float = 12.5, wrist: bool = True) -> Located:
    """What one frame located: ``objects``, by ``camera``, at shutter ``at``."""
    return Located(camera=camera, captured_at_s=at, mounting="eye_in_hand" if wrist else "eye_to_hand",
                   tool_to_base_mm=None, objects=tuple(objects))


class ScriptedLocator:
    """A locator over one camera that answers each prompt as the test scripts it.

    ``sees`` maps a prompt to the objects the camera locates for it, or to a callable of the arm's current TCP pose
    answering them (what a wrist camera sees depends on where the arm stands); a prompt it does not know locates
    nothing. ``raises`` makes the next locate raise. Every locate is written on ``asked`` and, with ``log``, on the
    shared log, so a test reads where the arm stood when it looked.
    """

    def __init__(self, sees: "Mapping[str, Any]", *, arm: Any = None, wrist: bool = True, rig_id: str = "wrist",
                 log: "list[Any] | None" = None, raises: "BaseException | None" = None,
                 raises_on: "Mapping[int, BaseException] | None" = None) -> None:
        self.sees = dict(sees)
        self.arm = arm
        self.on_the_wrist = wrist
        self.rig_id = rig_id
        self.log = log
        self.raises = raises
        self.raises_on = dict(raises_on or {})
        self.asked: list[str] = []
        self.count = 0

    def locate(self, prompt: str) -> Located:
        self.asked.append(prompt)
        if self.log is not None:
            self.log.append(("locate", prompt))
        if self.raises is not None:
            raising, self.raises = self.raises, None
            raise raising
        if len(self.asked) in self.raises_on:
            raise self.raises_on[len(self.asked)]
        self.count += 1
        answer = self.sees.get(prompt, ())
        if callable(answer):
            answer = answer(self.arm.get_tcp_pose() if self.arm is not None else None)
        return located(*tuple(answer), camera=self.rig_id, at=float(self.count), wrist=self.on_the_wrist)


# ---------------------------------------------------------------------------------------------------------------------
# One task, end to end, on the doubles
# ---------------------------------------------------------------------------------------------------------------------


#: The taught poses every task here knows, by name.
POSES = {"drop_left": PLACE_JOINTS, "park": PARK_JOINTS}


@dataclass
class Ran:
    """What one task did: its report, and the arm, the hand, the service, the hooks and the log it did it with."""

    report: Any
    log: list[Any]
    arm: TaskArm
    jaws: Any
    service: TaskService
    hooks: RecordingHooks

    def names(self) -> list[str]:
        return self.hooks.names()

    def after(self, name: str) -> list[Any]:
        """The log after the first event called ``name`` was recorded, read off the marks ``run`` leaves on it."""
        marks = [index for index, entry in enumerate(self.log) if entry == ("event", name)]
        return self.log[marks[0] + 1:] if marks else []


def run(picks: "Iterable[Pick | str]" = ("part",), *, place: Any = None, scope: str = "once",
        return_to: str = "home", options: Any = None, first_motion: str = "look", hooks: "RecordingHooks | None" = None,
        arm: "TaskArm | None" = None, jaws: Any = None, closed: bool = False, locators: Any = UNSET,
        poses: "Mapping[str, JointPositions] | None" = None, plan: Any = None, **service_keywords: Any) -> Ran:
    """Run one task on the doubles: ``picks`` scripted, a pose place at ``drop_left`` unless ``place`` says otherwise.

    Every event the task says is marked on the log as ``("event", name)``, so a test reads what the arm and the hand
    did after it.
    """
    from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan, run_task  # noqa: PLC0415

    log: list[Any] = arm.log if arm is not None else []
    arm = arm if arm is not None else TaskArm(log)
    jaws = jaws if jaws is not None else toggle(log, closed=closed, arm=arm)
    service = TaskService(arm, jaws, picks, **service_keywords)
    hooks = hooks if hooks is not None else RecordingHooks()
    recorded = hooks.on_event

    def mark(name: str, data: dict[str, Any]) -> None:
        log.append(("event", name))
        if recorded is not None:
            recorded(name, data)

    hooks.on_event = mark
    plan = plan if plan is not None else TaskPlan(
        object="red cube", place=place if place is not None else PlaceAt(pose="drop_left"), return_to=return_to,
        scope=scope, options=options if options is not None else TaskOptions(), first_motion=first_motion)
    keywords: dict[str, Any] = {} if not chosen(locators) else {"locators": locators}
    report = run_task(service, plan, hooks=hooks, poses=POSES if poses is None else poses, **keywords)
    return Ran(report=report, log=log, arm=arm, jaws=jaws, service=service, hooks=hooks)
