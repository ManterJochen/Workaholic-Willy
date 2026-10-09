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
* :class:`MeasuringLocator` is a wrist camera over a bench it renders (:class:`BinScene`): it grounds a bin's phrase from
  the pixels it sees of the bin, and reads a new frame where points should stand (``measure``), as ``Locator`` does.
* :class:`MeasuringHand` is a hand that is not a toggle: it closes and opens by intent and measures whether it holds a
  part, so a part that sticks after the jaws opened is still measured.

A bin is :func:`bin_points`: its four walls' tops at the rim, their inner faces and its floor, as a camera above sees
them, sampled every few millimetres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
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
    "BLUE",
    "BinScene",
    "GRASP_Z_MM",
    "HOME",
    "MeasuringHand",
    "MeasuringLocator",
    "SeenBin",
    "YELLOW",
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
    reports no grasp pose on a part picked. ``looks`` are the looks its report says it perceived from (none, as a pick
    handed no look reports), and ``targets_first_look`` how many targets its first look counted
    (``telemetry['targets_by_look']``, as a wrist pick reports them; ``None`` says nothing, as a fixed camera's pick).
    """

    kind: str = "part"
    xy: tuple[float, float] = PART_XY
    grasp: bool = True
    detector_failed: bool = False
    cloud: "np.ndarray | None" = None
    then: "Callable[[], None] | None" = None
    looks: tuple[str, ...] = ()
    targets_first_look: int | None = None


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
        if pick.looks or pick.targets_first_look is not None:
            counted = {} if pick.targets_first_look is None else {
                "targets_by_look": [pick.targets_first_look] + [0] * max(0, len(pick.looks) - 1)}
            report = replace(report, looks=tuple(pick.looks), telemetry={**dict(report.telemetry), **counted})
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
# A camera place's target, measured: a wrist camera over a bench it renders
# ---------------------------------------------------------------------------------------------------------------------

#: The plastic of the owner's two bins of one model, side by side on the cell, BGR.
YELLOW = (20, 200, 230)
BLUE = (200, 90, 20)
#: The bench, and what stands in front of a bin (a hand, a carried part), BGR.
_BENCH_BGR = (70, 70, 70)
_IN_FRONT_BGR = (30, 30, 30)
#: The rendered wrist camera: 320 x 240 with a 260 px focal length (2 mm a pixel at 520 mm), 200 mm behind the TCP on
#: the tool's axis and looking along it, so a tool pointing down looks straight down.
_WIDTH, _HEIGHT, _FOCAL = 320, 240, 260.0
_LENS = np.array([[_FOCAL, 0.0, _WIDTH / 2.0], [0.0, _FOCAL, _HEIGHT / 2.0], [0.0, 0.0, 1.0]])
_CAMERA_TO_TOOL = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -200.0], [0.0, 0.0, 0.0, 1.0]])


@dataclass
class SeenBin:
    """A bin on the bench: the phrase a detector grounds it by, its middle, its outer size, its rim, its colour (BGR)
    and the score the detector gives it. Its walls are 8 mm thick, its floor 5 mm."""

    label: str = "yellow bin"
    centre_xy: tuple[float, float] = BIN_CENTRE
    size_xy: tuple[float, float] = BIN_SIZE
    rim_mm: float = BIN_RIM_MM
    colour_bgr: tuple[int, int, int] = YELLOW
    score: float = 0.8


@dataclass
class BinScene:
    """What the bench holds, which a test changes between two looks: its bins, and upright boxes in front of them
    (``in_front``: a centre and half sizes, BASE mm), as a hand or a carried part hides a rim."""

    bins: list[SeenBin] = field(default_factory=lambda: [SeenBin()])
    in_front: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = field(default_factory=list)

    def move(self, dx: float, dy: float = 0.0, **changes: Any) -> None:
        """Move the first bin by ``dx``, ``dy``, changed as ``changes`` say (``size_xy``, ``rim_mm``, ...)."""
        first = self.bins[0]
        self.bins[0] = replace(first, centre_xy=(first.centre_xy[0] + dx, first.centre_xy[1] + dy), **changes)

    def take_away(self) -> None:
        self.bins = []

    def render(self, camera_to_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """What a camera at ``camera_to_base`` sees: the depth along its axis (mm, 0 where its ray meets nothing), the
        colour image (BGR), and which bin each pixel shows (-1: the bench, or what stands in front). Ray cast as
        ``tests/_seen_scenes.py`` casts, the bench at z = 0, a pixel's centre at its whole index."""
        from tests._seen_scenes import Solid, open_bin  # noqa: PLC0415

        cols, rows = np.meshgrid(np.arange(_WIDTH, dtype=np.float64), np.arange(_HEIGHT, dtype=np.float64))
        rays = np.stack([(cols - _LENS[0, 2]) / _FOCAL, (rows - _LENS[1, 2]) / _FOCAL, np.ones_like(cols)], axis=-1)
        rays = rays @ camera_to_base[:3, :3].T
        origin = camera_to_base[:3, 3]
        down = rays[..., 2] < 0.0
        depth = np.where(down, -origin[2] / np.where(down, rays[..., 2], -1.0), np.inf)
        shows = np.full(depth.shape, -1)
        solids = [(solid, index) for index, seen in enumerate(self.bins)
                  for solid in open_bin(seen.centre_xy, seen.size_xy, seen.rim_mm, wall_mm=8.0, floor_mm=5.0)]
        solids += [(Solid(centre, half), -2) for centre, half in self.in_front]
        for solid, index in solids:
            c, s = math.cos(math.radians(solid.yaw_deg)), math.sin(math.radians(solid.yaw_deg))
            start = origin - np.asarray(solid.centre, dtype=np.float64)
            start = np.array([c * start[0] + s * start[1], -s * start[0] + c * start[1], start[2]])
            way = np.stack([c * rays[..., 0] + s * rays[..., 1], -s * rays[..., 0] + c * rays[..., 1], rays[..., 2]],
                           axis=-1)
            half = np.asarray(solid.half, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                low, high = (-half - start) / way, (half - start) / way
            near = np.nanmax(np.minimum(low, high), axis=-1)
            far = np.nanmin(np.maximum(low, high), axis=-1)
            hit = (far >= near) & (near > 0.0) & (near < depth)
            depth = np.where(hit, near, depth)
            shows = np.where(hit, index, shows)
        image = np.empty((*depth.shape, 3), dtype=np.uint8)
        image[...] = _BENCH_BGR
        for index, seen in enumerate(self.bins):
            image[shows == index] = seen.colour_bgr
        image[shows == -2] = _IN_FRONT_BGR
        return np.where(np.isfinite(depth), depth, 0.0), image, np.where(shows == -2, -1, shows)


class MeasuringLocator:
    """A camera on the wrist over a bench it renders (:class:`BinScene`), 200 mm behind the TCP the arm reads now.

    ``locate`` grounds a phrase as a detector would: every bin of that phrase in a box of its own, its points the pixels
    the camera sees of it placed in BASE through the frame, the frame and its colour image kept, as a locator the
    service lends keeps them (``place_target.located_image``). ``measure`` reads a new frame where given points should
    stand, through ``Located.measure``, the code ``Locator.measure`` reads its own frame with. ``asked`` is every phrase
    located, ``measured`` how many points each measure read, and with ``log`` both go on the shared log;
    ``measure_raises`` makes the next measure raise.
    """

    on_the_wrist = True

    def __init__(self, scene: BinScene, *, arm: Any, log: "list[Any] | None" = None, rig_id: str = "wrist",
                 measure_raises: "BaseException | None" = None) -> None:
        from src.robot.execution import place_target  # noqa: PLC0415

        self.scene = scene
        self.arm = arm
        self.log = log
        self.rig_id = rig_id
        self.measure_raises = measure_raises
        self.asked: list[str] = []
        self.measured: list[int] = []
        self.count = 0
        self.sightings = place_target._Sightings(rig_id, keep=True)  # noqa: SLF001
        place_target._SIGHTINGS[self] = self.sightings  # noqa: SLF001

    def _taken(self) -> tuple[Any, np.ndarray, np.ndarray]:
        """One frame from where the arm stands: as ``Locator`` keeps it, its colour image (BGR), and what each pixel
        shows."""
        from src.robot.perception.locator import _Frame  # noqa: PLC0415 (the frame a locate keeps, as Locator keeps it)

        self.count += 1
        camera_to_base = np.asarray(self.arm.get_tcp_pose().to_matrix(), dtype=np.float64) @ _CAMERA_TO_TOOL
        depth, image, shows = self.scene.render(camera_to_base)
        return _Frame(depth_mm=depth, intrinsics=_LENS.copy(), camera_to_base=camera_to_base), image, shows

    def locate(self, prompt: str) -> Located:
        from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415

        self.asked.append(prompt)
        if self.log is not None:
            self.log.append(("locate", prompt))
        frame, image, shows = self._taken()
        objects: list[LocatedObject] = []
        for index, seen in enumerate(self.scene.bins):
            mask = (shows == index) & (frame.depth_mm > 0.0)
            if seen.label != prompt or not mask.any():
                continue
            points = to_base_mm(mask, frame.depth_mm, frame.intrinsics, frame.camera_to_base)
            objects.append(LocatedObject(label=prompt, score=seen.score, box_px=None, mask=mask, points_base_mm=points,
                                         centre_mm=tuple(float(v) for v in np.median(points, axis=0))))  # type: ignore[arg-type]
        located = Located(camera=self.rig_id, captured_at_s=float(self.count), mounting="eye_in_hand",
                          tool_to_base_mm=None, objects=tuple(objects), _frame=frame)
        self.sightings.show_located(located, image, prompt=prompt)
        return located

    def measure(self, points_base_mm: Any) -> Any:
        points = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
        self.measured.append(int(points.shape[0]))
        if self.log is not None:
            self.log.append(("measure", int(points.shape[0])))
        if self.measure_raises is not None:
            raising, self.measure_raises = self.measure_raises, None
            raise raising
        frame, image, _shows = self._taken()
        seen = Located(camera=self.rig_id, captured_at_s=float(self.count), mounting="eye_in_hand",
                       tool_to_base_mm=None, objects=(), _frame=frame)
        return seen.measure(points, image)


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
        poses: "Mapping[str, JointPositions] | None" = None, plan: Any = None, known_target: Any = None,
        **service_keywords: Any) -> Ran:
    """Run one task on the doubles: ``picks`` scripted, a pose place at ``drop_left`` unless ``place`` says otherwise,
    handed ``known_target`` where one is given.

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
    if known_target is not None:
        keywords["known_target"] = known_target
    report = run_task(service, plan, hooks=hooks, poses=POSES if poses is None else poses, **keywords)
    return Ran(report=report, log=log, arm=arm, jaws=jaws, service=service, hooks=hooks)


# ---------------------------------------------------------------------------------------------------------------------
# The places of a sort: a class list, located once
# ---------------------------------------------------------------------------------------------------------------------


class ClassListLocator(MeasuringLocator):
    """A :class:`MeasuringLocator` that grounds a class list as the cell's locator does once it is handed one (the
    sorting map's S1): ``"yellow bin | blue bin"`` grounds every bin of either phrase in one locate, each labelled with
    its own phrase, as the camera source labels a box once the locator maps the detector's words onto the list
    (``object_labels``); one phrase grounds as :class:`MeasuringLocator` does. ``called`` makes the detector call a bin
    by other words (a bin's own label to what it is called): asked for either, it grounds the bin under that name, as a
    detector that mistakes one bin for another, or gives a box a label of its own (``ambiguous``), does.
    """

    def __init__(self, scene: BinScene, *, arm: Any, log: "list[Any] | None" = None, rig_id: str = "wrist",
                 measure_raises: "BaseException | None" = None, called: "Mapping[str, str] | None" = None) -> None:
        super().__init__(scene, arm=arm, log=log, rig_id=rig_id, measure_raises=measure_raises)
        self.called = dict(called or {})

    def locate(self, prompt: str) -> Located:
        from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415

        self.asked.append(prompt)
        if self.log is not None:
            self.log.append(("locate", prompt))
        classes = [phrase.strip() for phrase in prompt.split("|")]
        frame, image, shows = self._taken()
        objects: list[LocatedObject] = []
        for index, seen in enumerate(self.scene.bins):
            mask = (shows == index) & (frame.depth_mm > 0.0)
            label = self.called.get(seen.label, seen.label)
            if (seen.label not in classes and label not in classes) or not mask.any():
                continue
            points = to_base_mm(mask, frame.depth_mm, frame.intrinsics, frame.camera_to_base)
            objects.append(LocatedObject(label=label, score=seen.score, box_px=None, mask=mask, points_base_mm=points,
                                         centre_mm=tuple(float(v) for v in np.median(points, axis=0))))  # type: ignore[arg-type]
        located = Located(camera=self.rig_id, captured_at_s=float(self.count), mounting="eye_in_hand",
                          tool_to_base_mm=None, objects=tuple(objects), _frame=frame)
        self.sightings.show_located(located, image, prompt=prompt)
        return located


__all__.append("ClassListLocator")


# ---------------------------------------------------------------------------------------------------------------------
# A sort: picks that say the kind they went for, a detector asked for one phrase, a cell that keeps the old rule
# ---------------------------------------------------------------------------------------------------------------------


def placing(arm: TaskArm, **place: Any) -> TaskArm:
    """``arm`` keeping a tree that says only how its parts are set down (``robot.place`` as ``place`` says, every other
    key its default): every other reading of the tree finds nothing there, as on an arm that keeps none.
    ``placing(arm, relocate=False)`` is a cell that keeps the old rule: a bin the check lost puts the part back."""
    from src.config.schema.robot.place_schema import RobotPlaceConfig  # noqa: PLC0415

    arm.config = SimpleNamespace(place=RobotPlaceConfig(**place))  # type: ignore[attr-defined]
    return arm


@dataclass
class SortPick(Pick):
    """One scripted pick of a sort, as a sort's pick loop reports it: ``label`` the kind of the part it went for (the
    label of the segmentation it gripped, ``PickReport.target_label``), ``unclaimed`` the labels its first look turned
    away that no rule claims (``PickReport.unclaimed_labels``: ``ambiguous``, a word no rule names)."""

    label: str = ""
    unclaimed: tuple[str, ...] = ()


class SortingService(TaskService):
    """``TaskService`` whose picks carry what a sort's pick loop says (:class:`SortPick`) on their reports."""

    def _play(self, pick: Pick) -> AutonomousGraspReport:
        report = super()._play(pick)
        label, unclaimed = str(getattr(pick, "label", "")), tuple(getattr(pick, "unclaimed", ()))
        if report.pick_report is None or not (label or unclaimed):
            return report
        said = SimpleNamespace(**vars(report.pick_report), target_label=label, unclaimed_labels=unclaimed)
        return replace(report, pick_report=said)


class OnePhraseLocator(ClassListLocator):
    """A :class:`ClassListLocator` that answers one phrase as a detector asked for one description does: every bin in
    view in a box of that phrase (a locate of one description reads every box as it). A class list still labels each
    bin with its own phrase, so only a class list tells two bins apart."""

    def locate(self, prompt: str) -> Located:
        if "|" in prompt:
            return super().locate(prompt)
        own, self.called = self.called, {seen.label: prompt.strip() for seen in self.scene.bins}
        try:
            return super().locate(prompt)
        finally:
            self.called = own


def run_sort(plan: Any, picks: "Iterable[Pick | str]" = (), *, hooks: "RecordingHooks | None" = None,
             arm: "TaskArm | None" = None, jaws: Any = None, locators: Any = UNSET,
             poses: "Mapping[str, JointPositions] | None" = None, known_target: Any = None, known_targets: Any = None,
             **service_keywords: Any) -> Ran:
    """Run ``plan`` (a sort, or a task of one kind) on the doubles with a :class:`SortingService`, handed
    ``known_target`` and ``known_targets`` where given; every event marked on the log as :func:`run` marks it."""
    from src.robot.execution.task import run_task  # noqa: PLC0415

    log: list[Any] = arm.log if arm is not None else []
    arm = arm if arm is not None else TaskArm(log)
    jaws = jaws if jaws is not None else toggle(log, arm=arm)
    service = SortingService(arm, jaws, picks, **service_keywords)
    hooks = hooks if hooks is not None else RecordingHooks()
    recorded = hooks.on_event

    def mark(name: str, data: dict[str, Any]) -> None:
        log.append(("event", name))
        if recorded is not None:
            recorded(name, data)

    hooks.on_event = mark
    keywords: dict[str, Any] = {} if not chosen(locators) else {"locators": locators}
    if known_target is not None:
        keywords["known_target"] = known_target
    if known_targets is not None:
        keywords["known_targets"] = known_targets
    report = run_task(service, plan, hooks=hooks, poses=POSES if poses is None else poses, **keywords)
    return Ran(report=report, log=log, arm=arm, jaws=jaws, service=service, hooks=hooks)


__all__.extend(["OnePhraseLocator", "SortPick", "SortingService", "placing", "run_sort"])
