"""A toggle hand pulses only where its owner expects it to, end to end, on every path the owner runs.

The owner's cell (2026-09-24): a UR10 CB3 with a Robotiq Hand-E on the Robotiq I/O Coupling, driven as ``jaw_io``
single_toggle on tool DO0, 24 V, no sensor, a 2 s stroke. Each rising edge on DO0 flips the jaws and nothing reads them
back, so the program counts its own pulses from where a person said the jaws stood when it connected. One pulse in the
wrong place inverts every command after it, and the whole cell fails there.

So the owner's own flows run here with the real ``JawIOGripper`` and the real ``Robot`` / ``handling`` /
``GraspExecutionPolicy`` / pick loop / pick service / ``PickRun``, on

* a Hand-E double (``_HandE``) on tool DO0 that flips its jaws on every rising edge, and so knows where they really
  stand whatever the program believes;
* a UR double (``_Arm``) that answers the controller's state, and refuses or protective-stops the motions a test names;
* a person at the terminal (``_Person``), answering through the gripper's ``ask=`` seam (the bench: through ``input``);

each writing to one log, every entry with its sequence number, so the order is read off the log and not off anybody's
bookkeeping. Every scenario is held to the same invariants (``_check``):

I1  no rising edge between the start of a pick and its first motion, unless a person answered 'p';
I2  at most one rising edge at the part per pick (tests say where it must be exactly one): after the line in has
    arrived and the controller was asked there, before the lift, with a ``close_settle_s`` wait between it and the lift;
I3  at most one rising edge per place, put back or release, where the count says closed and never where it says open,
    after the line in and before the line out, with the stroke waited out before the line out;
I4  after every verb the jaws stand where the program believes, or the program cannot say and asks before its next
    pulse, or the hand is disconnected and the next connect asks;
I5  the connect asks where the jaws stand exactly once per connect;
R6  a pick asks the person where the program believes the jaws closed (or cannot say), before any pulse or motion, and
    asks nothing where it believes them open; a release never asks.

The failures injected are the ones the owner's cell has met or can: a lift refused after the close, an approach
refused before it, a protective stop mid-pick, an I/O write that raised, and Ctrl-C right after a rising edge.
"""

from __future__ import annotations

import io
import unittest
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from src.geometry import Frame, Pose, Transform
from src.robot.core import (
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotError,
    RobotMode,
    RobotStatus,
    SafetyMode,
)
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
from src.robot.execution.handling import HandlingOutcome, HandOutcome
from src.robot.execution.lifecycle import connect_cell, disconnect_cell
from src.robot.execution.pick_run import PickOutcome, PickRun, PickRunReport, Recording
from src.robot.execution.robot import Robot
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
from src.robot.grasping.motion.grasp_motion import GraspMotion
from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.grippers.jaw_io import JawIOGripper
from tests.test_a_pick_looks_from_the_joints_its_program_declares import GRASP_MM, LOOK_A, LOOK_B, _Calculator
from tests.test_a_pick_looks_from_the_joints_its_program_declares import _Segment as _Mask

#: The owner's Hand-E: a 0.2 s pulse and a 2 s stroke both ways.
_PULSE_S = 0.2
_SETTLE_S = 2.0
#: What every motion of a known-part flow says instead of a camera world (example 05).
_BENCH = "a known part on a clear table, no camera"
#: Example 05's part and tray.
_PART = Pose.tool_down(450.0, 100.0, 120.0, yaw_deg=90.0)
_TRAY = Pose.tool_down(300.0, -250.0, 140.0)
#: Example 12: its looks, its motion (60 mm above the grasp, a 5 mm squeeze, a 100 mm lift), and where the scripted part
#: is grasped, BASE mm.
_LOOK = [LOOK_A, LOOK_B]
_MOTION = GraspMotion(standoff_mm=60.0, close_squeeze_mm=5.0, retreat_mm=100.0)
_GRASP = tuple(float(v) for v in GRASP_MM)
#: The words every "where do the jaws stand" question carries.
_WHERE = "Do they stand OPEN?"
_MOTIONS = ("move", "joints", "home")

_RUNNING = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                       emergency_stopped=False)
_PROTECTIVE = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP,
                          protective_stopped=True, emergency_stopped=False, message="Safetystatus: PROTECTIVE_STOP")


# ---------------------------------------------------------------------------------------------------
# The log and the doubles that write to it
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Event:
    seq: int
    kind: str
    data: tuple[Any, ...]

    def __repr__(self) -> str:
        return f"#{self.seq:<3} {self.kind:<8} {self.data}"


class _Log:
    """One log for the arm, the jaws, the clock and the person: an event's place on it is its sequence number."""

    def __init__(self) -> None:
        self.events: list[_Event] = []

    def write(self, kind: str, *data: Any) -> _Event:
        event = _Event(len(self.events), kind, tuple(data))
        self.events.append(event)
        return event

    def kinds(self, *kinds: str) -> list[_Event]:
        return [e for e in self.events if e.kind in kinds]

    def render(self) -> str:
        return "\n".join(repr(e) for e in self.events)


class _HandE:
    """Tool DO0 of the owner's UR10, wired to a Hand-E on the Robotiq I/O Coupling: every rising edge flips the jaws.

    ``closed`` is where the jaws really stand. ``next_high_fails`` makes the next HIGH write raise: ``lost`` before it
    reached the controller (nothing flips), ``reply`` after it did (the jaws flipped and the reply was lost), ``ctrl_c``
    a Ctrl-C inside the write after the controller took it.
    """

    def __init__(self, log: _Log, *, closed: bool = False) -> None:
        self.log = log
        self.closed = closed
        self.level = False
        self.next_high_fails = ""

    def set_digital_output(self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        if (int(pin), port) != (0, DigitalIOPort.TOOL):
            raise AssertionError(f"the owner's hand is on tool DO0, and {port.value} output {pin} was written")
        fails = ""
        if value:
            fails, self.next_high_fails = self.next_high_fails, ""
        if fails == "lost":
            self.log.write("DO lost", bool(value))
            raise RobotError("the controller did not take the write on tool output 0")
        rising = bool(value) and not self.level
        self.level = bool(value)
        self.log.write("DO", self.level)
        if rising:
            self.closed = not self.closed
            self.log.write("edge", "closed" if self.closed else "open")
        if fails == "reply":
            raise RobotError("the reply to the write on tool output 0 was lost")
        if fails == "ctrl_c":
            raise KeyboardInterrupt

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return self.level

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        raise AssertionError("a single toggle reads no input")

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        raise AssertionError("nothing analog is wired")


class _Clock:
    """The gripper's sleep: every wait on the log, and a Ctrl-C in the first wait after a rising edge once armed."""

    def __init__(self, log: _Log) -> None:
        self.log = log
        self.ctrl_c_after_edge = False

    def __call__(self, seconds: float) -> None:
        last = self.log.events[-1] if self.log.events else None
        self.log.write("sleep", float(seconds))
        if self.ctrl_c_after_edge and last is not None and last.kind == "edge":
            self.ctrl_c_after_edge = False
            raise KeyboardInterrupt


class _Person:
    """The person at the terminal: answers from a script, every question and answer on the log.

    A question nobody scripted is answered with end of input, which the driver takes as an abort, and is kept, so the
    invariants fail on it by name rather than on whatever the abort did next.
    """

    def __init__(self, log: _Log, *answers: str) -> None:
        self.log = log
        self.answers = list(answers)
        self.unscripted: list[str] = []

    def __call__(self, question: str) -> str:
        self.log.write("ask", question)
        if not self.answers:
            self.unscripted.append(question)
            raise EOFError
        answer = self.answers.pop(0)
        self.log.write("answer", answer)
        return answer


class _Arm(DummyRobotArm):
    """The owner's UR10 as far as the jaws care: every motion and every controller read on the log.

    Motions are counted by ``move`` call. ``refuse`` is the call the planner refuses on a running controller;
    ``stop_in`` the call that meets a protective stop; ``stop_after`` the call after whose arrival the controller
    protective-stops. A stopped controller refuses every motion after, joints included.
    """

    def __init__(self, log: _Log, *, refuse: "int | None" = None, stop_in: "int | None" = None,
                 stop_after: "int | None" = None) -> None:
        super().__init__()
        self.log = log
        self.refuse, self.stop_in, self.stop_after = refuse, stop_in, stop_after
        self.moves = 0
        self.status = _RUNNING

    def get_robot_status(self) -> RobotStatus:
        self.log.write("status", "running" if self.status.is_operational else "stopped")
        return self.status

    def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
        raise AssertionError("nothing on the owner's paths may clear a stop")

    def move(self, pose: Pose, **keywords: Any) -> MotionResult:
        index, self.moves = self.moves, self.moves + 1
        linear = bool(keywords.get("linear", False))
        where = tuple(round(float(v), 1) for v in pose.position_mm)
        if not self.status.is_operational or index == self.stop_in:
            self.status = _PROTECTIVE
            self.log.write("move", where, linear, "stopped")
            return MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                                       message="Driver reported moveL() failure")
        if index == self.refuse:
            self.log.write("move", where, linear, "refused")
            return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                                       message="rfinger|fixture:seen_00: mesh distance 0.4 mm")
        result = super().move(pose, **keywords)
        self.log.write("move", where, linear, "arrived")
        if index == self.stop_after:
            self.status = _PROTECTIVE
        return result

    def move_to_joints(self, joints: Any, **keywords: Any) -> MotionResult:
        if not self.status.is_operational:
            self.log.write("joints", joints.degrees(), "stopped")
            return MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_JOINTS,
                                       target_joints=joints, message="Driver reported moveJ() failure")
        result = super().move_to_joints(joints, **keywords)
        self.log.write("joints", joints.degrees(), "arrived")
        return result

    def move_home(self) -> bool:
        if not self.status.is_operational:
            self.log.write("home", "stopped")
            return False
        self.log.write("home", "arrived")
        return super().move_home()


class _Camera:
    """The wrist D415 as the pick loop reads it: the part in view or not, scripted per frame, each frame on the log."""

    def __init__(self, log: _Log, sees: "list[bool]") -> None:
        self.log = log
        self.sees = list(sees)

    def acquire(self) -> PerceptionFrame:
        seen = self.sees.pop(0) if len(self.sees) > 1 else self.sees[0]
        self.log.write("perceive", seen)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[11:22, 11:22] = 1
        return PerceptionFrame(depth_map=np.full((32, 32), 500.0),
                               intrinsics=np.array([[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]]),
                               segmentations=(_Mask(mask),) if seen else ())


@dataclass
class _Cell:
    log: _Log
    hand_e: _HandE
    clock: _Clock
    person: _Person
    jaws: JawIOGripper
    arm: _Arm


def _cell(*answers: str, closed: bool = False, **arm: Any) -> _Cell:
    """The owner's cell: the real driver on the Hand-E double, the person answering ``answers`` in turn."""
    log = _Log()
    hand_e = _HandE(log, closed=closed)
    clock = _Clock(log)
    person = _Person(log, *answers)
    jaws = JawIOGripper(hand_e, actuation="single_toggle", close_output_pin=0, io_port="tool", pulse_s=_PULSE_S,
                        close_settle_s=_SETTLE_S, min_width_mm=5.0, max_width_mm=49.99, ask=person, sleep=clock)
    return _Cell(log, hand_e, clock, person, jaws, _Arm(log, **arm))


def _belief(jaws: JawIOGripper) -> str:
    """Where the program believes the jaws stand: its count, or ``unknown`` after a pulse whose high write failed."""
    if jaws._edge_unknown:
        return "unknown"
    return "closed" if jaws.jaws_closed else "open"


@contextmanager
def _verb(cell: _Cell, what: str) -> Iterator[None]:
    """One verb of the program on the log: begun with the belief, ended with the belief, the jaws and the link."""
    cell.log.write("begin", what, _belief(cell.jaws))
    raised = ""
    try:
        yield
    except BaseException as exc:
        raised = type(exc).__name__
        raise
    finally:
        cell.log.write("end", what, _belief(cell.jaws), "closed" if cell.hand_e.closed else "open",
                       cell.jaws.is_connected, raised)


def _robot(cell: _Cell) -> Robot:
    return Robot.from_parts(arm=cell.arm, gripper=cell.jaws, lock_key=None)


@contextmanager
def _connected(cell: _Cell, robot: Robot) -> Iterator[Any]:
    """``with robot.connected():`` as the examples write it, its connect marked on the log as one verb."""
    session = robot.connected()
    with _verb(cell, "connect"):
        session.__enter__()
    try:
        yield session
    finally:
        session.__exit__(None, None, None)


@contextmanager
def _cell_connected(cell: _Cell) -> Iterator[None]:
    """The cell brought up and down as ``Cell.connected()`` does it (``connect_cell``), the connect marked as a verb."""
    with _verb(cell, "connect"):
        connect_cell(cell.arm, cell.jaws)
    try:
        yield
    finally:
        disconnect_cell(cell.arm, cell.jaws)


def _pick(cell: _Cell, robot: Robot, pose: Pose, width_mm: float) -> Any:
    with _verb(cell, "pick"):
        return robot.pick(pose, width_mm, decline=_BENCH)


def _place(cell: _Cell, robot: Robot, pose: Pose) -> Any:
    with _verb(cell, "place"):
        return robot.place(pose, decline=_BENCH)


def _release(cell: _Cell, robot: Robot) -> Any:
    with _verb(cell, "release"):
        return robot.release()


class _Watched:
    """The pick service, each pick and each put back marked on the log as one verb."""

    def __init__(self, service: Any, cell: _Cell) -> None:
        self.service = service
        self.cell = cell

    def pick(self, **keywords: Any) -> Any:
        with _verb(self.cell, "pick"):
            return self.service.pick(**keywords)

    def put_back(self, report: Any) -> Any:
        with _verb(self.cell, "put_back"):
            return self.service.put_back(report)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.service, name)


def _service(cell: _Cell, sees: "tuple[bool, ...]" = (True,)) -> AutonomousGraspService:
    """Example 12's pick service on the owner's cell: a wrist camera, its motion, the toggle hand."""
    wrist = EyeInHandFrameResolver(t_cam_to_tool=Transform.from_matrix(np.eye(4), from_frame=Frame.CAMERA,
                                                                       to_frame=Frame.TOOL))
    return AutonomousGraspService.from_components(
        arm=cell.arm, calculator=_Calculator(),  # type: ignore[arg-type]
        perception=_Camera(cell.log, list(sees)), mode=GraspMode.EASY, gripper=cell.jaws, frame_resolver=wrist,
        max_attempts=1, motion=_MOTION)


def _campaign(cell: _Cell, service: Any, runs: int, **keywords: Any) -> PickRunReport:
    """Example 12's campaign: the program's looks, and each lifted part put back where it was grasped."""
    return PickRun.from_service(_Watched(service, cell), runs=runs, recording=Recording.off(), look=_LOOK,
                                put_back=True, **keywords).execute()


# ---------------------------------------------------------------------------------------------------
# The invariants
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Span:
    """One verb as the log holds it."""

    what: str
    begin: _Event
    body: tuple[_Event, ...]
    end: _Event

    @property
    def believed_at_start(self) -> str:
        return str(self.begin.data[1])

    @property
    def believed(self) -> str:
        return str(self.end.data[1])

    @property
    def actual(self) -> str:
        return str(self.end.data[2])

    @property
    def connected(self) -> bool:
        return bool(self.end.data[3])

    @property
    def raised(self) -> str:
        return str(self.end.data[4])

    def of(self, *kinds: str) -> list[_Event]:
        return [e for e in self.body if e.kind in kinds]


def _spans(log: _Log) -> list[_Span]:
    spans: list[_Span] = []
    opened: "_Event | None" = None
    body: list[_Event] = []
    for e in log.events:
        if e.kind == "begin":
            assert opened is None, f"verbs nested at #{e.seq}"
            opened, body = e, []
        elif e.kind == "end":
            assert opened is not None and opened.data[0] == e.data[0], f"an end with no begin at #{e.seq}"
            spans.append(_Span(str(e.data[0]), opened, tuple(body), e))
            opened = None
        elif opened is not None:
            body.append(e)
    assert opened is None, "a verb never ended"
    return spans


def _at_the_part(t: unittest.TestCase, s: _Span, edge: _Event, at: str) -> None:
    """I2 / I3: ``edge`` came where the line in arrived, the controller asked there, and the stroke before leaving."""
    motions = s.of(*_MOTIONS)
    before = [m for m in motions if m.seq < edge.seq]
    t.assertTrue(before, f"a pulse before any motion of the {s.what}: {at}")
    line = before[-1]
    t.assertEqual(("move", True, "arrived"), (line.kind, line.data[1], line.data[2]),
                  f"the pulse #{edge.seq} did not follow a line in that arrived: {at}")
    t.assertTrue(any(e.kind == "status" and e.data[0] == "running" and line.seq < e.seq < edge.seq for e in s.body),
                 f"the controller was not asked at the part before the pulse #{edge.seq}: {at}")
    after = [m for m in motions if m.seq > edge.seq]
    settle = [e.seq for e in s.body if e.kind == "sleep" and e.data[0] == _SETTLE_S and e.seq > edge.seq]
    if after:
        out = after[0]
        t.assertEqual(("move", True), (out.kind, out.data[1]), f"the motion after the pulse is not a line: {at}")
        t.assertGreater(out.data[0][2], line.data[0][2], f"the motion after the pulse is not up and out: {at}")
        t.assertTrue(any(seq < out.seq for seq in settle),
                     f"no close_settle_s between the pulse #{edge.seq} and the motion after it: {at}")
    elif not s.raised and s.believed != "unknown":
        # A verb that ended where it pulsed waits the stroke before it returns. Not after a write that raised: the
        # driver cannot say the jaws moved, and refuses its next pulse until a person has looked at them.
        t.assertTrue(settle, f"no close_settle_s after the pulse #{edge.seq}: {at}")


def _check(t: unittest.TestCase, cell: _Cell) -> list[_Span]:
    """Hold the whole log to I1 to I5 and R6; the verbs, for a test's own assertions."""
    trail = cell.log.render()
    spans = _spans(cell.log)
    inside = {e.seq for s in spans for e in s.body}
    t.assertEqual([], [e for e in cell.log.kinds("edge") if e.seq not in inside],
                  f"a rising edge outside every verb:\n{trail}")
    lost = False
    for s in spans:
        at = f"{s.what} #{s.begin.seq}..#{s.end.seq}\n{trail}"
        edges, motions, asks, answers = s.of("edge"), s.of(*_MOTIONS), s.of("ask"), s.of("answer")
        chosen = [a for a in answers if str(a.data[0]).strip().lower() in ("p", "pulse")]
        if lost:
            # R8: the last verb left the jaws unknown; the next pulse only after a person has said where they stand.
            for e in edges:
                t.assertTrue(any(a.seq < e.seq for a in answers),
                             f"R8: pulse #{e.seq} on jaws nobody can name, with nobody asked: {at}")
        first = motions[0].seq if motions else None
        if s.what in ("connect", "pick"):
            # I1: before the first motion, a pulse only for a person's 'p', one each.
            early = [e for e in edges if first is None or e.seq < first]
            t.assertLessEqual(len(early), len(chosen), f"I1: a pulse before the arm moved that nobody chose: {at}")
            for edge, answer in zip(early, chosen):
                t.assertLess(answer.seq, edge.seq, f"I1: pulse #{edge.seq} before the person chose it: {at}")
        if s.what == "connect":
            t.assertEqual(1, sum(1 for a in asks if _WHERE in str(a.data[0])), f"I5: the connect asked not once: {at}")
            t.assertEqual([], motions, f"the connect moved the arm: {at}")
        elif s.what == "pick":
            if s.believed_at_start == "open":
                t.assertEqual([], asks, f"R6: a pick on jaws believed open asked: {at}")
            elif edges or motions:
                t.assertTrue(asks and asks[0].seq < min(x.seq for x in edges + motions),
                             f"R6: a pick on jaws believed {s.believed_at_start} did not ask first: {at}")
            closes = [e for e in edges if first is not None and e.seq > first]
            t.assertLessEqual(len(closes), 1, f"I2: more than one pulse at the part: {at}")
            for e in closes:
                _at_the_part(t, s, e, at)
        elif s.what == "move":
            t.assertEqual([], edges, f"a motion pulsed the jaws: {at}")
        else:  # place, put_back, release, grasp
            t.assertEqual([], asks, f"a {s.what} asked the person: {at}")
            t.assertLessEqual(len(edges), 1, f"I3: more than one pulse in one {s.what}: {at}")
            already = "closed" if s.what == "grasp" else "open"
            if s.believed_at_start == already:
                t.assertEqual([], edges, f"a {s.what} on jaws believed {already} pulsed: {at}")
            for e in edges:
                if motions:
                    _at_the_part(t, s, e, at)
                elif not s.raised and s.believed != "unknown":
                    t.assertTrue(any(x.kind == "sleep" and x.data[0] == _SETTLE_S and x.seq > e.seq for x in s.body),
                                 f"no close_settle_s after the pulse #{e.seq}: {at}")
        # I4: the jaws stand where the program believes, or it cannot say (and asks), or the hand is disconnected.
        t.assertTrue(s.believed in (s.actual, "unknown") or not s.connected,
                     f"I4: after the {s.what} the program believes {s.believed} and the jaws stand {s.actual}: {at}")
        lost = s.connected and s.believed == "unknown"
    t.assertEqual([], cell.person.unscripted, f"questions nobody scripted:\n{trail}")
    return spans


def _only(spans: list[_Span], what: str) -> _Span:
    found = [s for s in spans if s.what == what]
    assert len(found) == 1, f"{len(found)} {what} verb(s)"
    return found[0]


def _line_before(s: _Span, edge: _Event) -> _Event:
    return [m for m in s.of(*_MOTIONS) if m.seq < edge.seq][-1]


def _at(pose: Pose) -> tuple[float, ...]:
    return tuple(round(float(v), 1) for v in pose.position_mm)


# ---------------------------------------------------------------------------------------------------
# The connect question (R1, I5)
# ---------------------------------------------------------------------------------------------------


class TheConnectAsksOnceTests(unittest.TestCase):
    def test_open_sends_nothing_and_the_count_starts_open(self) -> None:
        cell = _cell("")
        with _connected(cell, _robot(cell)):
            pass
        _check(self, cell)
        self.assertEqual([], cell.log.kinds("edge"))
        self.assertEqual(1, len(cell.log.kinds("ask")))

    def test_closed_and_p_is_one_pulse_before_anything_moves_and_example_05_runs_from_there(self) -> None:
        cell = _cell("closed", "p", closed=True)
        robot = _robot(cell)
        with _connected(cell, robot):
            picked = _pick(cell, robot, _PART, 40.0)
            placed = _place(cell, robot, _TRAY)
        self.assertTrue(picked.ok and placed.ok, f"{picked.render()}\n{placed.render()}")
        spans = _check(self, cell)
        connect = _only(spans, "connect")
        self.assertEqual(["open"], [e.data[0] for e in connect.of("edge")])
        self.assertLess(connect.of("edge")[0].seq, cell.log.kinds(*_MOTIONS)[0].seq)
        self.assertEqual(3, len(cell.log.kinds("edge")), "the chosen pulse, the close and the open")
        self.assertFalse(cell.hand_e.closed)
        self.assertEqual([], cell.person.answers)

    def test_closed_and_a_refuses_the_connect_with_nothing_pulsed_and_the_arm_rolled_back(self) -> None:
        cell = _cell("closed", "a", closed=True)
        with self.assertRaises(RobotError):
            with _connected(cell, _robot(cell)):
                self.fail("connected on jaws a person said stand closed and chose not to open")
        _check(self, cell)
        self.assertEqual([], cell.log.kinds("edge", *_MOTIONS))
        self.assertFalse(cell.arm.is_connected, "the arm was not rolled back")
        self.assertFalse(cell.jaws.is_connected)
        self.assertTrue(cell.hand_e.closed)

    def test_every_connect_asks_once_and_counts_from_the_answer_not_from_before(self) -> None:
        """A program ends holding the part; the next connect of the same hand asks, and counts from the answer."""
        cell = _cell("", "closed", "p")
        robot = _robot(cell)
        with _connected(cell, robot):
            self.assertTrue(_pick(cell, robot, _PART, 40.0).ok)
        with _connected(cell, robot):
            self.assertTrue(_place(cell, robot, _TRAY).outcome is HandlingOutcome.EXECUTED)
        spans = _check(self, cell)
        self.assertEqual(2, len([s for s in spans if s.what == "connect"]))
        second = [s for s in spans if s.what == "connect"][1]
        self.assertEqual(1, len(second.of("edge")), "the second connect did not open the jaws the person said closed")
        self.assertEqual([], _only(spans, "place").of("edge"), "the place pulsed jaws the count says stand open")
        self.assertFalse(cell.hand_e.closed)


# ---------------------------------------------------------------------------------------------------
# The owner's flows, nothing failing
# ---------------------------------------------------------------------------------------------------


class Example05PicksAndPlacesAKnownPartTests(unittest.TestCase):
    def test_one_pulse_at_the_part_and_one_at_the_tray_whatever_the_width(self) -> None:
        for width in (3.0, 40.0, 49.99):  # R7: no width decides a pulse
            with self.subTest(width=width):
                cell = _cell("")
                robot = _robot(cell)
                with _connected(cell, robot):
                    picked = _pick(cell, robot, _PART, width)
                    self.assertIs(HandlingOutcome.EXECUTED, picked.outcome, picked.render())
                    placed = _place(cell, robot, _TRAY)
                    self.assertIs(HandlingOutcome.EXECUTED, placed.outcome, placed.render())
                spans = _check(self, cell)
                pick, place = _only(spans, "pick"), _only(spans, "place")
                self.assertEqual(["closed"], [e.data[0] for e in pick.of("edge")])
                self.assertEqual(_at(_PART), _line_before(pick, pick.of("edge")[0]).data[0])
                self.assertEqual(["open"], [e.data[0] for e in place.of("edge")])
                self.assertEqual(_at(_TRAY), _line_before(place, place.of("edge")[0]).data[0])
                self.assertEqual(2, len(cell.log.kinds("edge")))
                self.assertFalse(cell.hand_e.closed)

    def test_a_release_on_open_jaws_sends_nothing(self) -> None:
        cell = _cell("")
        robot = _robot(cell)
        with _connected(cell, robot):
            released = _release(cell, robot)
        self.assertIs(HandOutcome.RELEASED, released.outcome, released.render())
        self.assertEqual("already open: no pulse", released.note)
        _check(self, cell)
        self.assertEqual([], cell.log.kinds("edge"))

    def test_example_04_open_close_open_is_two_pulses(self) -> None:
        cell = _cell("")
        robot = _robot(cell)
        with _connected(cell, robot):
            _release(cell, robot)
            with _verb(cell, "grasp"):
                held = robot.grasp(39.0)
            robot.is_holding()
            _release(cell, robot)
        self.assertIs(HandOutcome.GRASPED, held.outcome, held.render())
        _check(self, cell)
        self.assertEqual(["closed", "open"], [e.data[0] for e in cell.log.kinds("edge")])

    def test_example_13_hands_over_with_one_pulse_after_the_move(self) -> None:
        cell = _cell("")
        robot = _robot(cell)
        with _connected(cell, robot):
            self.assertTrue(_pick(cell, robot, _PART, 40.0).ok)
            with _verb(cell, "move"):
                self.assertTrue(robot.move(Pose.tool_down(200.0, -400.0, 320.0), decline=_BENCH).ok)
            released = _release(cell, robot)
        self.assertIs(HandOutcome.RELEASED, released.outcome, released.render())
        spans = _check(self, cell)
        self.assertEqual(1, len(_only(spans, "release").of("edge")))
        self.assertLess(_only(spans, "move").end.seq, _only(spans, "release").of("edge")[0].seq)
        self.assertFalse(cell.hand_e.closed)


class Example12CampaignTests(unittest.TestCase):
    def test_three_runs_from_the_looks_each_pulse_once_at_the_part_and_once_at_the_put_back(self) -> None:
        cell = _cell("")
        with _cell_connected(cell):
            run = _campaign(cell, _service(cell, sees=(False, True)), 3)
        self.assertEqual(3, run.succeeded, run.render())
        self.assertEqual(0, run.exit_code)
        spans = _check(self, cell)
        picks = [s for s in spans if s.what == "pick"]
        backs = [s for s in spans if s.what == "put_back"]
        self.assertEqual((3, 3), (len(picks), len(backs)))
        for pick in picks:
            self.assertEqual(["closed"], [e.data[0] for e in pick.of("edge")])
            self.assertEqual("joints", pick.of(*_MOTIONS)[0].kind, "the pick did not start at its look")
            self.assertEqual(_GRASP, _line_before(pick, pick.of("edge")[0]).data[0])
        for back in backs:
            self.assertEqual(["open"], [e.data[0] for e in back.of("edge")])
            self.assertEqual(_GRASP, _line_before(back, back.of("edge")[0]).data[0])
        self.assertEqual(2, len(picks[0].of("joints")), "the first pick looked from LOOK_A and then LOOK_B")
        self.assertEqual(6, len(cell.log.kinds("edge")))
        self.assertEqual(1, len(cell.log.kinds("ask")), "asked beyond the connect")
        self.assertFalse(cell.hand_e.closed)

    def test_a_pre_open_width_in_the_motion_sends_no_pulse_before_the_look(self) -> None:
        cell = _cell("")
        service = AutonomousGraspService.from_components(
            arm=cell.arm, calculator=_Calculator(),  # type: ignore[arg-type]
            perception=_Camera(cell.log, [True]), mode=GraspMode.EASY, gripper=cell.jaws,
            frame_resolver=EyeInHandFrameResolver(t_cam_to_tool=Transform.from_matrix(
                np.eye(4), from_frame=Frame.CAMERA, to_frame=Frame.TOOL)),
            max_attempts=1, motion=GraspMotion(standoff_mm=60.0, pre_open_width_mm=49.99))
        with _cell_connected(cell):
            run = _campaign(cell, service, 2)
        self.assertEqual(2, run.succeeded, run.render())
        _check(self, cell)
        self.assertEqual(4, len(cell.log.kinds("edge")))


# ---------------------------------------------------------------------------------------------------
# A lift refused after the close
# ---------------------------------------------------------------------------------------------------


class ALiftRefusedAfterTheCloseTests(unittest.TestCase):
    def test_example_05_keeps_the_count_closed_and_the_next_pick_asks_before_anything(self) -> None:
        cell = _cell("", "closed", "a", refuse=2)
        robot = _robot(cell)
        with _connected(cell, robot):
            picked = _pick(cell, robot, _PART, 40.0)
            self.assertIs(HandlingOutcome.MOTION_REFUSED, picked.outcome, picked.render())
            again = _pick(cell, robot, _PART, 40.0)
        self.assertIs(HandlingOutcome.REFUSED, again.outcome, again.render())
        self.assertIn("not starting the pick", again.message)
        spans = _check(self, cell)
        first, second = [s for s in spans if s.what == "pick"]
        self.assertEqual(["closed"], [e.data[0] for e in first.of("edge")])
        self.assertEqual("refused", first.of(*_MOTIONS)[-1].data[2])
        self.assertEqual(([], []), (second.of("edge"), second.of(*_MOTIONS)))
        self.assertTrue(cell.hand_e.closed and cell.jaws.jaws_closed)

    def test_a_campaign_goes_on_and_its_next_pick_asks_before_its_look(self) -> None:
        """Run 1's lift is refused with the part in the jaws; run 2 asks, the person opens them, and it runs."""
        cell = _cell("", "closed", "p", refuse=2)
        with _cell_connected(cell):
            run = _campaign(cell, _service(cell), 3)
        self.assertEqual([PickOutcome.FAILED, PickOutcome.SUCCEEDED, PickOutcome.SUCCEEDED],
                         [a.outcome for a in run.attempts], run.render())
        self.assertIsNone(run.attempts[0].put_back)
        spans = _check(self, cell)
        picks = [s for s in spans if s.what == "pick"]
        self.assertEqual(["closed"], [e.data[0] for e in picks[0].of("edge")])
        self.assertEqual(["open", "closed"], [e.data[0] for e in picks[1].of("edge")])
        self.assertLess(picks[1].of("edge")[0].seq, picks[1].of(*_MOTIONS)[0].seq)
        self.assertEqual(6, len(cell.log.kinds("edge")))
        self.assertFalse(cell.hand_e.closed)
        self.assertEqual([], cell.person.answers)


# ---------------------------------------------------------------------------------------------------
# An approach refused before the close
# ---------------------------------------------------------------------------------------------------


class AnApproachRefusedBeforeTheCloseTests(unittest.TestCase):
    def test_example_05_sends_nothing_and_the_next_pick_runs_without_asking(self) -> None:
        cell = _cell("", refuse=1)
        robot = _robot(cell)
        with _connected(cell, robot):
            refused = _pick(cell, robot, _PART, 40.0)
            self.assertIs(HandlingOutcome.MOTION_REFUSED, refused.outcome, refused.render())
            self.assertTrue(_pick(cell, robot, _PART, 40.0).ok)
        spans = _check(self, cell)
        first, second = [s for s in spans if s.what == "pick"]
        self.assertEqual([], first.of("edge"))
        self.assertEqual(1, len(second.of("edge")))
        self.assertEqual(1, len(cell.log.kinds("ask")))

    def test_a_campaign_whose_first_standoff_is_refused_goes_on_with_nothing_asked(self) -> None:
        cell = _cell("", refuse=0)
        with _cell_connected(cell):
            run = _campaign(cell, _service(cell), 3)
        self.assertEqual([PickOutcome.FAILED, PickOutcome.SUCCEEDED, PickOutcome.SUCCEEDED],
                         [a.outcome for a in run.attempts], run.render())
        spans = _check(self, cell)
        self.assertEqual([], [s for s in spans if s.what == "pick"][0].of("edge"))
        self.assertEqual(4, len(cell.log.kinds("edge")))
        self.assertEqual(1, len(cell.log.kinds("ask")))


# ---------------------------------------------------------------------------------------------------
# A protective stop mid-pick
# ---------------------------------------------------------------------------------------------------


class AProtectiveStopMidPickTests(unittest.TestCase):
    def test_a_stop_between_the_line_in_and_the_close_closes_nothing(self) -> None:
        cell = _cell("", stop_after=1)
        robot = _robot(cell)
        with _connected(cell, robot):
            picked = _pick(cell, robot, _PART, 40.0)
        self.assertIs(HandlingOutcome.REFUSED, picked.outcome, picked.render())
        assert picked.hand is not None
        self.assertIn("protective_stop=True", picked.hand.error)
        _check(self, cell)
        self.assertEqual([], cell.log.kinds("edge"))
        self.assertFalse(cell.hand_e.closed)

    def test_a_stop_in_the_lift_keeps_the_part_and_nothing_releases_it_on_the_stopped_arm(self) -> None:
        cell = _cell("", stop_in=2)
        robot = _robot(cell)
        with _connected(cell, robot):
            picked = _pick(cell, robot, _PART, 40.0)
            placed = _place(cell, robot, _TRAY)
            released = _release(cell, robot)
            again = _pick(cell, robot, _PART, 40.0)
        self.assertIs(HandlingOutcome.MOTION_REFUSED, picked.outcome, picked.render())
        self.assertIs(HandlingOutcome.REFUSED, placed.outcome, placed.render())
        self.assertIs(HandOutcome.REFUSED, released.outcome, released.render())
        self.assertIs(HandlingOutcome.REFUSED, again.outcome, again.render())
        spans = _check(self, cell)
        self.assertEqual(["closed"], [e.data[0] for e in cell.log.kinds("edge")])
        self.assertEqual([], _only(spans, "place").of(*_MOTIONS))
        self.assertEqual([], [s for s in spans if s.what == "pick"][1].of("ask"),
                         "Robot.pick asked the hand on a stopped controller")
        self.assertTrue(cell.hand_e.closed and cell.jaws.jaws_closed)

    def test_a_campaign_stopped_in_a_lift_stops_with_the_part_held(self) -> None:
        cell = _cell("", stop_in=8)  # run 1: moves 0 to 5 (pick and put back); run 2: standoff 6, line 7, lift 8
        with _cell_connected(cell):
            run = _campaign(cell, _service(cell), 3)
        self.assertEqual([PickOutcome.SUCCEEDED, PickOutcome.RAISED, PickOutcome.CANCELLED],
                         [a.outcome for a in run.attempts], run.render())
        self.assertIn("the controller cannot move", run.attempts[1].detail)
        spans = _check(self, cell)
        self.assertEqual(1, len([s for s in spans if s.what == "put_back"]))
        self.assertEqual(["closed", "open", "closed"], [e.data[0] for e in cell.log.kinds("edge")])
        self.assertTrue(cell.hand_e.closed and cell.jaws.jaws_closed)

    def test_a_campaign_stopped_between_the_line_in_and_the_close_closes_nothing(self) -> None:
        cell = _cell("", stop_after=1)
        with _cell_connected(cell):
            run = _campaign(cell, _service(cell), 2)
        self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts],
                         run.render())
        _check(self, cell)
        self.assertEqual([], cell.log.kinds("edge"))

    def test_a_later_campaign_on_the_stopped_arm_neither_asks_nor_pulses(self) -> None:
        """The console keeps its cell across runs, and a program may start a second campaign: on an arm still stopped
        with the part in the jaws, the pick ends on the controller, as ``Robot.pick`` does (the test above), before the
        hand is asked anything. A pulse there drops the part from wherever the lift stopped.

        Red before: ``AutonomousGraspService.pick`` asked the toggle where its jaws stand (``_hand_refusal``) before
        anything asked the controller, so the person was asked, answered closed, chose 'p', and DO0 pulsed on the
        stopped arm. The person's answers stay scripted here, so a regression shows as that pulse and not as a
        question nobody scripted."""
        cell = _cell("", "closed", "p", stop_in=8)
        service = _service(cell)
        with _cell_connected(cell):
            first = _campaign(cell, service, 3)
            later = _campaign(cell, service, 1)
        self.assertIs(PickOutcome.RAISED, first.attempts[1].outcome, first.render())
        spans = _check(self, cell)
        last = [s for s in spans if s.what == "pick"][-1]
        self.assertEqual([], last.of("ask"), "the hand was asked on a stopped controller")
        self.assertEqual([], last.of("edge"), "a pulse on the stopped arm dropped the part")
        self.assertEqual([], last.of(*_MOTIONS, "perceive"), "the later pick moved or perceived on a stopped arm")
        self.assertEqual(["stopped"], [e.data[0] for e in last.of("status")], "the controller was not asked first")
        self.assertIs(PickOutcome.RAISED, later.attempts[0].outcome, later.render())
        self.assertIn("the controller cannot move", later.attempts[0].detail)
        assert later.last is not None
        self.assertTrue(later.last.controller_stopped, later.last.render())
        self.assertEqual(["closed", "p"], cell.person.answers, "the person was asked on the stopped arm")
        self.assertTrue(cell.hand_e.closed and cell.jaws.jaws_closed)


# ---------------------------------------------------------------------------------------------------
# An I/O write that raised
# ---------------------------------------------------------------------------------------------------


class AnIOWriteThatRaisedTests(unittest.TestCase):
    def test_a_high_that_never_arrived_stops_every_pulse_until_a_person_says(self) -> None:
        cell = _cell("", "open")
        cell.hand_e.next_high_fails = "lost"
        robot = _robot(cell)
        with _connected(cell, robot):
            picked = _pick(cell, robot, _PART, 40.0)
            released = _release(cell, robot)
            again = _pick(cell, robot, _PART, 40.0)
        self.assertIs(HandlingOutcome.GRIPPER_FAULT, picked.outcome, picked.render())
        self.assertIs(HandOutcome.GRIPPER_FAULT, released.outcome, released.render())
        self.assertTrue(again.ok, again.render())
        spans = _check(self, cell)
        first = [s for s in spans if s.what == "pick"][0]
        self.assertEqual("unknown", first.believed)
        self.assertEqual([], first.of("edge"))
        self.assertEqual(2, len(first.of(*_MOTIONS)), "the arm lifted after a close nobody can vouch for")
        self.assertEqual(["closed"], [e.data[0] for e in cell.log.kinds("edge")])

    def test_a_high_whose_reply_was_lost_is_asked_about_and_opened_by_the_persons_pulse(self) -> None:
        cell = _cell("", "closed", "p")
        cell.hand_e.next_high_fails = "reply"
        robot = _robot(cell)
        with _connected(cell, robot):
            picked = _pick(cell, robot, _PART, 40.0)
            again = _pick(cell, robot, _PART, 40.0)
        self.assertIs(HandlingOutcome.GRIPPER_FAULT, picked.outcome, picked.render())
        self.assertTrue(again.ok, again.render())
        spans = _check(self, cell)
        first, second = [s for s in spans if s.what == "pick"]
        self.assertEqual(("unknown", "closed"), (first.believed, first.actual))
        self.assertEqual(["open", "closed"], [e.data[0] for e in second.of("edge")])
        self.assertFalse(cell.hand_e.level, "DO0 left high")

    def test_a_campaign_stops_on_it_and_the_next_pick_asks(self) -> None:
        cell = _cell("", "closed", "p")
        cell.hand_e.next_high_fails = "reply"
        service = _service(cell)
        with _cell_connected(cell):
            run = _campaign(cell, service, 3)
            later = _campaign(cell, service, 1)
        self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED, PickOutcome.CANCELLED],
                         [a.outcome for a in run.attempts], run.render())
        self.assertIn("the gripper needs a person", run.attempts[0].detail)
        self.assertEqual(1, later.succeeded, later.render())
        spans = _check(self, cell)
        self.assertEqual(1, len([s for s in spans if s.what == "put_back"]), "a put back after the fault")
        self.assertEqual(["closed", "open", "closed", "open"], [e.data[0] for e in cell.log.kinds("edge")])
        self.assertFalse(cell.hand_e.closed)


# ---------------------------------------------------------------------------------------------------
# Ctrl-C right after a rising edge
# ---------------------------------------------------------------------------------------------------


class CtrlCRightAfterARisingEdgeTests(unittest.TestCase):
    def test_in_the_pulse_after_the_close_the_count_agrees_and_the_cell_comes_down(self) -> None:
        cell = _cell("")
        cell.clock.ctrl_c_after_edge = True
        robot = _robot(cell)
        with self.assertRaises(KeyboardInterrupt):
            with _connected(cell, robot):
                _pick(cell, robot, _PART, 40.0)
        spans = _check(self, cell)
        pick = _only(spans, "pick")
        self.assertEqual(("closed", "closed", "KeyboardInterrupt"), (pick.believed, pick.actual, pick.raised))
        self.assertEqual([], [m for m in pick.of(*_MOTIONS) if m.seq > pick.of("edge")[0].seq], "the lift ran")
        self.assertFalse(cell.hand_e.level, "DO0 left high")
        self.assertFalse(cell.jaws.is_connected or cell.arm.is_connected, "the cell did not come down")

    def test_inside_the_high_write_the_program_stops_and_the_next_connect_asks(self) -> None:
        cell = _cell("", "closed", "p")
        cell.hand_e.next_high_fails = "ctrl_c"
        robot = _robot(cell)
        with self.assertRaises(KeyboardInterrupt):
            with _connected(cell, robot):
                _pick(cell, robot, _PART, 40.0)
        with _connected(cell, robot):
            self.assertTrue(_pick(cell, robot, _PART, 40.0).ok)
        spans = _check(self, cell)
        first = [s for s in spans if s.what == "pick"][0]
        self.assertEqual(("unknown", "closed"), (first.believed, first.actual))
        self.assertFalse(cell.jaws.is_connected)
        self.assertEqual(["closed", "open", "closed"], [e.data[0] for e in cell.log.kinds("edge")])
        self.assertTrue(cell.hand_e.closed)

    def test_in_the_pulse_after_a_place_opens_the_count_agrees(self) -> None:
        cell = _cell("")
        robot = _robot(cell)
        with self.assertRaises(KeyboardInterrupt):
            with _connected(cell, robot):
                self.assertTrue(_pick(cell, robot, _PART, 40.0).ok)
                cell.clock.ctrl_c_after_edge = True
                _place(cell, robot, _TRAY)
        spans = _check(self, cell)
        place = _only(spans, "place")
        self.assertEqual(("open", "open"), (place.believed, place.actual))
        self.assertEqual(2, len(place.of(*_MOTIONS)), "the line out ran after Ctrl-C")

    def test_in_a_campaign_it_leaves_the_campaign_with_the_count_agreeing(self) -> None:
        cell = _cell("")

        def _arm_ctrl_c(attempt: Any) -> None:
            if attempt.index == 0:
                cell.clock.ctrl_c_after_edge = True

        with _cell_connected(cell):
            with self.assertRaises(KeyboardInterrupt):
                _campaign(cell, _service(cell), 3, on_attempt=_arm_ctrl_c)
        spans = _check(self, cell)
        last = [s for s in spans if s.what == "pick"][-1]
        self.assertEqual(("closed", "closed", "KeyboardInterrupt"), (last.believed, last.actual, last.raised))
        self.assertEqual(1, len([s for s in spans if s.what == "put_back"]))
        self.assertEqual(3, len(cell.log.kinds("edge")))


# ---------------------------------------------------------------------------------------------------
# The bench
# ---------------------------------------------------------------------------------------------------


class _BenchUR(_HandE):
    """What ``create_arm`` hands the bench on the owner's cell: tool DO0 on the Hand-E double, and a connect."""

    def connect(self) -> None:
        self.log.write("arm", "connect")

    def disconnect(self) -> None:
        self.log.write("arm", "disconnect")


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


class TheBenchJawsTests(unittest.TestCase):
    """``python -m src.robot.drivers.ur --jaws open|closed``, built from config, the person answering at ``input``."""

    def _bench(self, jaws: str, *answers: str, closed: bool) -> tuple[int, str, _Log, _BenchUR, _Person]:
        from src.config.schema.robot import RobotConfig
        from src.robot.drivers.ur import __main__ as bench_cli

        log = _Log()
        ur = _BenchUR(log, closed=closed)
        person = _Person(log, *answers)
        cfg = RobotConfig.model_validate({
            "vendor": "ur", "gripper": {"vendor": "jaw_io", "min_width_mm": 5.0, "max_width_mm": 49.99,
                                        "jaw_io": {"actuation": "single_toggle", "close_output_pin": 0,
                                                   "io_port": "tool", "close_settle_s": 0.0, "pulse_s": 0.05}}})
        out = io.StringIO()
        with patch.object(bench_cli, "_load_robot_config", lambda *_a: cfg), \
                patch("src.robot.drivers.create_arm", lambda *_a, **_k: ur), \
                patch.dict("os.environ", {"WILLY_PROFILE": ""}), \
                patch("sys.stdin", _Terminal("")), patch("builtins.input", person), redirect_stdout(out):
            code = bench_cli.main(["--jaws", jaws, "--yes"])
        return code, out.getvalue(), log, ur, person

    def test_the_jaws_end_where_the_command_says_with_one_question(self) -> None:
        #       --jaws    answers           jaws before  exit  edges  jaws after
        table = [("closed", ("",), False, 0, 1, True),
                 ("open", ("",), False, 0, 0, False),
                 ("open", ("closed", "p"), True, 0, 1, False),
                 ("closed", ("closed", "p"), True, 0, 2, True),
                 ("closed", ("closed", "a"), True, 1, 0, True)]
        for jaws, answers, before, code_wanted, edges, after in table:
            with self.subTest(jaws=jaws, answers=answers):
                code, printed, log, ur, person = self._bench(jaws, *answers, closed=before)
                self.assertEqual(code_wanted, code, printed)
                self.assertEqual(edges, len(log.kinds("edge")), log.render())
                self.assertEqual(after, ur.closed)
                self.assertFalse(ur.level, "DO0 left high")
                self.assertEqual(1, sum(1 for e in log.kinds("ask") if _WHERE in e.data[0]))
                self.assertEqual(([], []), (person.answers, person.unscripted))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
