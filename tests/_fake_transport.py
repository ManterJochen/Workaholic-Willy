"""A transport with no controller behind it, for a ``PlannedArm`` built with no vendor subclass (P0g slice 5).

One mixin per Protocol of ``robot.core.transport``, each with real members (Python 3.12 checks a runtime Protocol with
``getattr_static``, so a member a mixin only inherits from a Protocol stub is no member), composed into a class that
offers exactly the Protocols asked for: ``fake_transport()`` offers all twelve, which is what ``URTransport`` offers,
and ``fake_transport(offers=(ArmTransport,))`` a transport with no controller kinematics, no Cartesian route, no
payload register, no model string, no halt latch, no I/O, no force readings, no status, no stop release and no
provenance, which is less than a Franka or KUKA transport will offer and exactly what every transport must.

What it answers:

- Every call is on the tape, in order, with its arguments as given and unrounded (``tape``; ``calls(verb)``), and
  every waypoint the controller took is in ``sent``.
- A scripted verb answers from its queue (``script(verb, *answers)``): an answer is returned, an exception
  instance is raised, and a queue that runs dry raises :class:`ScriptExhausted`, so a test never reads a default it
  did not know it was reading. ``respond(verb, answer_for)`` answers each call from what it was asked, and
  ``always(verb, answer)`` answers every call the same, as a controller standing still reads the same each time. A
  queued script answers first, then a responder, then an answer given always. A verb nobody scripted answers its
  default, stated at each verb.
- The controller's IK finds no solution unless scripted (``None``, the Protocol's answer for it), and the
  kinematics are scripted unless a chain is handed in: ``kinematics=chain_of("ur5e")`` answers FK from the arm's
  description, with the controller's TCP register composed on the flange (``tcp_register``, identity by default,
  which is a controller running a bare flange).
- A trajectory is taken whole by default; ``Rejects(at=k)`` answers ``False`` with waypoint ``k`` sent and none
  after it, and ``Raises(at=k, error)`` raises with waypoint ``k`` sent, except for a ``ValueError``, which means the
  controller's library refused that waypoint before sending it (``ArmTransport``'s contract, P0g R3).
- The halt latch (``HoldsAHaltLatch``, Willy's own): ``request_halt`` latches it and a second request keeps the first
  record; while it is set no trajectory and no Cartesian move is sent (``last_move_end`` is ``REFUSED_HALTED``) and no
  output is written (``ArmHalted``) but the end of a pulse, as the UR connection refuses them, until ``clear_halt``.
  A read of the latch is the link's own state, asked before every motion, and stays off the tape.

The behaviour it shares with ``URTransport`` is held by ``tests/test_every_transport_keeps_the_transport_contract.py``,
and its members by ``tests/test_the_fake_transport_keeps_step_with_the_protocols.py``; ``tests/_injection_guard.py``
refuses a write that would create a name on it.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.kinematics import DhRow, SerialChain
from src.robot.core import (
    ControllerPayload,
    ControllerProvenance,
    MotionStatus,
    RobotMode,
    RobotStatus,
    SafetyMode,
    Wrench,
)
from src.robot.core.arm_capabilities import ArmHalted, DigitalIOPort, HaltState, MoveEnd, halted_refusal
from src.robot.core.transport import (
    ArmTransport,
    ControllerFreedrive,
    ControllerIO,
    ControllerKinematics,
    ControllerMotion,
    HoldsAHaltLatch,
    ReadsControllerProvenance,
    ReadsForceTorque,
    ReleasesProtectiveStop,
    ReportsControllerModel,
    ReportsControllerStatus,
    TakesPayload,
    WatchesController,
)

__all__ = [
    "Call",
    "silenced",
    "NotScripted",
    "Raises",
    "Rejects",
    "ScriptExhausted",
    "chain_of",
    "fake_transport",
]

#: A UR's home, a joint vector every six-joint description can stand at.
_HOME = (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)

#: The two properties, which answer the fake's state, and what sets that state instead of a script.
_READ_FROM_STATE = {
    "is_connected": "build it with connected=, or call connect and disconnect",
    "last_reject_status": "script move_pose with Rejects(status=...)",
}


#: Whether every fake has stopped recording: the sabotage control's switch (``silenced``), never set by a test itself.
_SILENCED = [False]


@contextmanager
def silenced() -> Iterator[None]:
    """Every fake answers as before and records nothing, neither its tape nor what it sent (P0g slice 13).

    The sabotage control's instrument: a test that reads what a fake was sent must fail inside this block, or its
    negatives would pass on a fake that recorded nothing (``tests/test_the_kit_tests_read_what_the_fake_records.py``).
    """
    _SILENCED[0] = True
    try:
        yield
    finally:
        _SILENCED[0] = False


class ScriptExhausted(AssertionError):
    """A scripted verb was asked once more than the test scripted it: the test would read an answer it never gave."""


class NotScripted(AssertionError):
    """A verb with no default was asked and never scripted: kinematics with no chain handed in."""


@dataclass(frozen=True)
class Call:
    """One call on the transport: the verb, its positional and its keyword arguments, as they were passed."""

    verb: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Rejects:
    """A controller that says no: ``run_trajectory`` answers ``False`` at waypoint ``at``, ``move_pose`` with
    ``status`` as its ``last_reject_status``, ``None`` a controller that refused and gave no cause."""

    at: int = 0
    status: MotionStatus | None = MotionStatus.CONTROLLER_REJECTED


@dataclass(frozen=True)
class Raises:
    """A controller that raises ``error`` at waypoint ``at`` of a trajectory."""

    error: BaseException
    at: int = 0


def chain_of(model: str) -> SerialChain:
    """The serial chain the arm registry describes for ``model``, for ``kinematics=``."""
    from src.config.arms import load_arm

    joints = load_arm(model).joints
    return SerialChain.from_dh(
        tuple(DhRow(a_m=j.a_m, d_m=j.d_m, alpha_rad=j.alpha_rad) for j in joints),
        joint_names=tuple(j.name for j in joints),
    )


class _State:
    """What every fake holds: the tape, the scripts, the joints and whether it is connected."""

    def __init__(
        self,
        *,
        joints: Sequence[float] = _HOME,
        connected: bool = True,
        kinematics: SerialChain | None = None,
        tcp_register: np.ndarray | None = None,
        model: str | None = None,
    ) -> None:
        self.tape: list[Call] = []
        self.sent: list[list[float]] = []
        self.joints: list[float] = [float(v) for v in joints]
        self.connected = bool(connected)
        #: A watch is open (``connect_watch``): status and release answer, and ``connected`` stays as it is.
        self.watching = False
        self.kinematics = kinematics
        self.tcp_register = np.eye(4) if tcp_register is None else np.asarray(tcp_register, dtype=np.float64)
        self.model = model
        self.rejected: MotionStatus | None = None
        #: The halt latch while it is set, how many times it was set, and how the last move asked for ended.
        self.halt: HaltState | None = None
        self.halts = 0
        self.move_end = MoveEnd.NONE
        #: The I/O registers by (bank, pin): what was written to an output, and what an input reads (False unset).
        self.outputs: dict[tuple[str, int], Any] = {}
        self.inputs: dict[tuple[str, int], bool] = {}
        self._scripts: dict[str, deque[Any]] = {}
        self._always: dict[str, Any] = {}
        self._responders: dict[str, Any] = {}

    def respond(self, verb: str, answer_for: Any) -> None:
        """Answer ``verb`` with ``answer_for(*args, **kwargs)``, called with what the call was asked, on every call no
        script is queued for: a controller whose answer depends on the question (the FK of the joints asked, a moveJ
        refused at one station). What it returns is an answer as a script's would be, an exception instance raised."""
        if verb in _READ_FROM_STATE:
            raise AttributeError(f"{verb} is read from the fake's state, never from a script: {_READ_FROM_STATE[verb]}")
        if not callable(getattr(type(self), verb, None)):
            raise AttributeError(f"this fake does not offer {verb!r}, so an answer for it would never be read")
        self._responders[verb] = answer_for

    def always(self, verb: str, answer: Any) -> None:
        """Answer ``verb`` with ``answer`` on every call no script is queued for: a controller standing still reads
        the same every time. An exception instance is raised each time. A queued script still answers first."""
        if verb in _READ_FROM_STATE:
            raise AttributeError(f"{verb} is read from the fake's state, never from a script: {_READ_FROM_STATE[verb]}")
        if not callable(getattr(type(self), verb, None)):
            raise AttributeError(f"this fake does not offer {verb!r}, so an answer for it would never be read")
        self._always[verb] = answer

    def script(self, verb: str, *answers: Any) -> None:
        """Queue ``answers`` for ``verb``, in order; once scripted, a verb answers only from its queue."""
        if verb in _READ_FROM_STATE:
            raise AttributeError(f"{verb} is read from the fake's state, never from a script: {_READ_FROM_STATE[verb]}")
        if not callable(getattr(type(self), verb, None)):
            raise AttributeError(f"this fake does not offer {verb!r}, so a script for it would never be read")
        self._scripts.setdefault(verb, deque()).extend(answers)

    def calls(self, verb: str) -> list[Call]:
        """Every call of ``verb``, in order."""
        return [call for call in self.tape if call.verb == verb]

    def _record(self, verb: str, *args: Any, **kwargs: Any) -> None:
        if _SILENCED[0]:
            return
        self.tape.append(Call(verb, tuple(args), dict(kwargs)))

    def _took(self, waypoint: list[float]) -> None:
        if not _SILENCED[0]:
            self.sent.append(waypoint)

    def _scripted(self, verb: str) -> tuple[bool, Any]:
        """``(True, answer)`` from the queue, raising an exception answer; ``(False, None)`` for an unscripted verb."""
        queue = self._scripts.get(verb)
        if queue:
            answer = queue.popleft()
        elif verb in self._responders:
            asked = self.tape[-1]  # every verb records before it asks for its answer
            assert asked.verb == verb, (asked.verb, verb)
            answer = self._responders[verb](*asked.args, **asked.kwargs)
        elif verb in self._always:
            answer = self._always[verb]
        elif queue is None:
            return False, None
        else:
            raise ScriptExhausted(f"{verb} was scripted and asked once more than its script answers")
        if isinstance(answer, BaseException):
            raise answer
        return True, answer

    def _flange_register(self, joints: Sequence[float], verb: str) -> np.ndarray:
        """The controller's TCP register at ``joints``: the chain's flange with ``tcp_register`` on it."""
        if self.kinematics is None:
            raise NotScripted(f"{verb} has no kinematics to answer from: script it, or hand in kinematics=chain_of(...)")
        links = self.kinematics.link_transforms_mm(np.asarray(joints, dtype=np.float64))
        if links is None:
            raise ValueError(f"{verb}: {len(joints)} joint(s) for a {self.kinematics.dof}-joint chain")
        return links[-1] @ self.tcp_register

    def _pose_answer(self, verb: str, joints: Sequence[float], label: str) -> Pose:
        scripted, answer = self._scripted(verb)
        if scripted:
            pose = answer if isinstance(answer, Pose) else Pose.from_matrix(np.asarray(answer), frame=Frame.BASE)
            return Pose(position_mm=pose.position_mm, quaternion_xyzw=pose.quaternion_xyzw, frame=Frame.BASE,
                        label=label)
        return Pose.from_matrix(self._flange_register(joints, verb), frame=Frame.BASE, label=label)


class _ArmTransport:
    """``ArmTransport``: every planned arm's transport."""

    @property
    def is_connected(self: Any) -> bool:
        """Whether it is connected: ``connected=`` at build, then ``connect`` and ``disconnect``."""
        return bool(self.connected)

    def connect(self: Any) -> None:
        """Connects, unless scripted to raise."""
        self._record("connect")
        self._scripted("connect")
        self.connected = True

    def disconnect(self: Any) -> None:
        self._record("disconnect")
        self._scripted("disconnect")
        self.connected = False
        self.watching = False

    def read_joints(self: Any) -> list[float]:
        """Where the arm stands: ``joints=`` at build, then the last waypoint a trajectory ran to."""
        self._record("read_joints")
        scripted, answer = self._scripted("read_joints")
        return [float(v) for v in answer] if scripted else list(self.joints)

    def run_trajectory(self: Any, waypoints: Sequence[Sequence[float]], *, vel: float | None,
                       acc: float | None) -> bool:
        """Every waypoint taken by default; ``True``, ``False``, :class:`Rejects` or :class:`Raises` when scripted."""
        path = [[float(v) for v in waypoint] for waypoint in waypoints]
        self._record("run_trajectory", path, vel=vel, acc=acc)
        if self.halt is not None:
            self.move_end = MoveEnd.REFUSED_HALTED
            return False
        self.move_end = MoveEnd.SENT
        scripted, answer = self._scripted("run_trajectory")
        stop_at, error = len(path), None
        if scripted:
            if answer is False:
                stop_at = 0
            elif isinstance(answer, Rejects):
                stop_at = answer.at
            elif isinstance(answer, Raises):
                stop_at, error = answer.at, answer.error
            elif answer is not True:
                raise TypeError(f"run_trajectory answers True, False, Rejects or Raises, not {answer!r}")
            if (answer is not True) and stop_at >= len(path):
                raise AssertionError(f"scripted to stop at waypoint {stop_at} of a {len(path)}-waypoint trajectory")
        for index, waypoint in enumerate(path):
            if index == stop_at:
                if not isinstance(error, ValueError):
                    self._took(waypoint)
                if error is not None:
                    raise error
                return False
            self._took(waypoint)
            self.joints = list(waypoint)
        return True

    def stop(self: Any) -> None:
        self._record("stop")
        self._scripted("stop")

    def wait_until_steady(self: Any, timeout_s: float, poll_interval_s: float) -> bool:
        """At rest (``True``) unless scripted."""
        self._record("wait_until_steady", timeout_s, poll_interval_s)
        scripted, answer = self._scripted("wait_until_steady")
        return bool(answer) if scripted else True


class _ControllerKinematics:
    """``ControllerKinematics``: a controller that solves its own kinematics and reports its TCP register."""

    def is_steady(self: Any) -> bool:
        """At rest (``True``) unless scripted."""
        self._record("is_steady")
        scripted, answer = self._scripted("is_steady")
        return bool(answer) if scripted else True

    def read_tcp(self: Any, *, label: str = "") -> Pose:
        """The TCP register at the joints it stands at, from the chain, or scripted."""
        self._record("read_tcp", label=label)
        return self._pose_answer("read_tcp", self.joints, label)

    def controller_tcp_now(self: Any) -> np.ndarray:
        """The TCP register at this instant as a 4x4 in mm, from the chain, or scripted."""
        self._record("controller_tcp_now")
        scripted, answer = self._scripted("controller_tcp_now")
        if scripted:
            return np.asarray(answer, dtype=np.float64)
        return self._flange_register(self.joints, "controller_tcp_now")

    def controller_fk(self: Any, joints: Sequence[float], *, label: str = "") -> Pose:
        """The TCP register at ``joints``, from the chain, or scripted."""
        self._record("controller_fk", [float(v) for v in joints], label=label)
        return self._pose_answer("controller_fk", joints, label)

    def controller_ik(self: Any, pose: Pose, *, seed: Sequence[float] | None = None) -> list[float] | None:
        """No solution (``None``) unless scripted."""
        self._record("controller_ik", pose, seed=None if seed is None else [float(v) for v in seed])
        scripted, answer = self._scripted("controller_ik")
        if not scripted or answer is None:
            return None
        return [float(v) for v in answer]


class _ControllerMotion:
    """``ControllerMotion``: a controller that moves to a Cartesian pose after its own IK."""

    def move_pose(self: Any, pose: Pose, *, linear: bool, vel: float | None, acc: float | None, register: bool,
                  workspace_pose: Pose) -> bool:
        """Taken (``True``) by default; ``False`` or :class:`Rejects` (with its status) when scripted."""
        self._record("move_pose", pose, linear=linear, vel=vel, acc=acc, register=register,
                     workspace_pose=workspace_pose)
        if self.halt is not None:
            self.rejected, self.move_end = MotionStatus.CANCELLED, MoveEnd.REFUSED_HALTED
            return False
        self.rejected, self.move_end = None, MoveEnd.SENT
        scripted, answer = self._scripted("move_pose")
        if not scripted or answer is True:
            return True
        self.rejected = answer.status if isinstance(answer, Rejects) else MotionStatus.CONTROLLER_REJECTED
        return False

    @property
    def last_reject_status(self: Any) -> MotionStatus | None:
        """Why the last ``move_pose`` answered ``False``, or ``None``."""
        return self.rejected  # type: ignore[no-any-return]


class _TakesPayload:
    """``TakesPayload``: a controller told the carried mass and its centre of gravity."""

    def set_payload(self: Any, mass_kg: float, cog_mm: Sequence[float]) -> None:
        self._record("set_payload", mass_kg, [float(v) for v in cog_mm])
        self._scripted("set_payload")


class _ReportsControllerModel:
    """``ReportsControllerModel``: a controller that says which arm it is."""

    def controller_model(self: Any) -> str | None:
        """``model=`` at build, unless scripted."""
        self._record("controller_model")
        scripted, answer = self._scripted("controller_model")
        return answer if scripted else self.model  # type: ignore[no-any-return]


class _ControllerIO:
    """``ControllerIO``: the controller's I/O registers, by bank."""

    def write_digital(self: Any, pin: int, value: bool, *, port: DigitalIOPort) -> bool:
        """Taken (``True``) and written to the register unless scripted; a ``False`` answer leaves it unwritten. Refused
        while latched, as the UR connection refuses it (``ArmHalted``, nothing written)."""
        self._record("write_digital", pin, value, port=port)
        if self.halt is not None:
            raise ArmHalted(halted_refusal(self.halt.reason, f"{port} digital output {pin} was not switched, and "
                                                               "nothing was sent"))
        scripted, answer = self._scripted("write_digital")
        taken = bool(answer) if scripted else True
        if taken:
            self.outputs[(str(port), int(pin))] = bool(value)
        return taken

    def read_digital_input(self: Any, pin: int, *, port: DigitalIOPort) -> bool:
        """The input register (``inputs``, ``False`` unset), unless scripted."""
        self._record("read_digital_input", pin, port=port)
        scripted, answer = self._scripted("read_digital_input")
        return answer if scripted else self.inputs.get((str(port), int(pin)), False)  # type: ignore[no-any-return]

    def read_digital_output(self: Any, pin: int, *, port: DigitalIOPort) -> bool:
        """The level last written to the output (``False`` unwritten), unless scripted."""
        self._record("read_digital_output", pin, port=port)
        scripted, answer = self._scripted("read_digital_output")
        return answer if scripted else bool(self.outputs.get((str(port), int(pin)), False))  # type: ignore[no-any-return]

    def write_analog(self: Any, pin: int, value: float, *, current: bool) -> bool:
        """Taken (``True``) unless scripted; refused while latched."""
        self._record("write_analog", pin, value, current=current)
        if self.halt is not None:
            raise ArmHalted(halted_refusal(self.halt.reason, f"analog output {pin} was not switched, and nothing "
                                                               "was sent"))
        scripted, answer = self._scripted("write_analog")
        taken = bool(answer) if scripted else True
        if taken:
            self.outputs[("analog", int(pin))] = (float(value), bool(current))
        return taken

    def end_pulse(self: Any, pin: int, *, port: DigitalIOPort) -> bool:
        """The output driven low, latched or not, unless scripted: the one write a halted link makes."""
        self._record("end_pulse", pin, port=port)
        scripted, answer = self._scripted("end_pulse")
        taken = bool(answer) if scripted else True
        if taken:
            self.outputs[(str(port), int(pin))] = False
        return taken


class _HoldsAHaltLatch:
    """``HoldsAHaltLatch`` (Willy's own): the latch the console's halt sets on the link to the controller."""

    def request_halt(self: Any, reason: str) -> HaltState:
        """Latches, sending nothing; a second request keeps the first record, as the UR connection does."""
        self._record("request_halt", reason)
        if self.halt is None:
            self.halt = HaltState(reason=str(reason).strip() or "halt requested", requested_at=time.time())
            self.halts += 1
        return self.halt  # type: ignore[no-any-return]

    def clear_halt(self: Any) -> None:
        self._record("clear_halt")
        self.halt = None

    def halt_state(self: Any) -> HaltState | None:
        """The latch while it is set, else ``None``; off the tape, since it asks the controller nothing."""
        return self.halt  # type: ignore[no-any-return]

    @property
    def brake_on_halt(self: Any) -> bool:
        """``False``: a move in flight runs to its end, as on an arm that does not brake (every UR brakes)."""
        return False

    @property
    def last_move_end(self: Any) -> MoveEnd | None:
        """How the last trajectory or Cartesian move asked for ended."""
        return self.move_end  # type: ignore[no-any-return]

    @property
    def halt_requests(self: Any) -> int | None:
        """How many times the latch was set; a clear does not count, nor a second request."""
        return self.halts  # type: ignore[no-any-return]


#: What a controller that says nothing unusual reports: running, normal, no stop latched, the program running.
READY = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                    emergency_stopped=False, message="", program_running=True)


class _ReadsForceTorque:
    """``ReadsForceTorque``: the wrench at the TCP and the joint torques."""

    def read_tcp_wrench(self: Any) -> Wrench:
        """No force (all zero, base frame) unless scripted."""
        self._record("read_tcp_wrench")
        scripted, answer = self._scripted("read_tcp_wrench")
        return answer if scripted else Wrench(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, frame=Frame.BASE)  # type: ignore[no-any-return]

    def read_joint_torques(self: Any) -> tuple[float, ...]:
        """No torque at any joint unless scripted."""
        self._record("read_joint_torques")
        scripted, answer = self._scripted("read_joint_torques")
        return tuple(float(t) for t in answer) if scripted else tuple(0.0 for _ in self.joints)


class _ReportsControllerStatus:
    """``ReportsControllerStatus``: the robot and safety mode, the stops and the controller's text."""

    def read_status(self: Any) -> RobotStatus:
        """:data:`READY` unless scripted."""
        self._record("read_status")
        scripted, answer = self._scripted("read_status")
        return answer if scripted else READY  # type: ignore[no-any-return]

    def read_status_quickly(self: Any) -> RobotStatus:
        """:data:`READY` with no text (Willy's own quick read), unless scripted."""
        self._record("read_status_quickly")
        scripted, answer = self._scripted("read_status_quickly")
        return answer if scripted else dataclasses.replace(READY, message="")  # type: ignore[no-any-return]


class _ReleasesProtectiveStop:
    """``ReleasesProtectiveStop``: a person releases a protective stop from software."""

    def release_protective_stop(self: Any) -> bool:
        """Released (``True``) unless scripted."""
        self._record("release_protective_stop")
        scripted, answer = self._scripted("release_protective_stop")
        return answer if scripted else True  # type: ignore[no-any-return]


class _WatchesController:
    """``WatchesController``: a link that reads the status and releases a stop and starts no control program (P0i)."""

    @property
    def is_watching(self: Any) -> bool:
        """Whether a watch is open: ``connect_watch``, then ``disconnect``."""
        return bool(self.watching)

    def connect_watch(self: Any) -> None:
        """Opens the watch, unless scripted to raise; a no-op while connected, as the UR connection's is."""
        self._record("connect_watch")
        self._scripted("connect_watch")
        if not self.connected:
            self.watching = True


class _ReadsControllerProvenance:
    """``ReadsControllerProvenance``: what the controller says it is."""

    def read_provenance(self: Any) -> ControllerProvenance:
        """A provenance that says nothing unless scripted: no fake proves a simulator by default."""
        self._record("read_provenance")
        scripted, answer = self._scripted("read_provenance")
        return answer if scripted else ControllerProvenance()  # type: ignore[no-any-return]


class _ControllerFreedrive:
    """``ControllerFreedrive``: a controller a person may move the arm on, behind a watchdog (port S4)."""

    def arm_watchdog(self: Any, min_frequency_hz: float) -> bool:
        """Armed (``True``) unless scripted."""
        self._record("arm_watchdog", min_frequency_hz)
        scripted, answer = self._scripted("arm_watchdog")
        return answer if scripted else True  # type: ignore[no-any-return]

    def kick_watchdog(self: Any) -> bool:
        """Answered (``True``) unless scripted."""
        self._record("kick_watchdog")
        scripted, answer = self._scripted("kick_watchdog")
        return answer if scripted else True  # type: ignore[no-any-return]

    def begin_hand_guiding(self: Any) -> bool:
        """Taken (``True``) unless scripted."""
        self._record("begin_hand_guiding")
        scripted, answer = self._scripted("begin_hand_guiding")
        return answer if scripted else True  # type: ignore[no-any-return]

    def end_hand_guiding(self: Any) -> bool:
        """Taken (``True``) unless scripted."""
        self._record("end_hand_guiding")
        scripted, answer = self._scripted("end_hand_guiding")
        return answer if scripted else True  # type: ignore[no-any-return]

    def restart_control_program(self: Any) -> None:
        """A fresh program comes up unless an exception is scripted."""
        self._record("restart_control_program")
        self._scripted("restart_control_program")

    def control_program_running(self: Any) -> bool:
        """Running (``True``) unless scripted."""
        self._record("control_program_running")
        scripted, answer = self._scripted("control_program_running")
        return answer if scripted else True  # type: ignore[no-any-return]

    def stop_latched(self: Any) -> bool:
        """No stop (``False``) unless scripted."""
        self._record("stop_latched")
        scripted, answer = self._scripted("stop_latched")
        return answer if scripted else False  # type: ignore[no-any-return]

    def read_joint_speeds(self: Any) -> list[float]:
        """Standing still unless scripted."""
        self._record("read_joint_speeds")
        scripted, answer = self._scripted("read_joint_speeds")
        return [float(v) for v in answer] if scripted else [0.0 for _ in self.joints]

    def read_payload(self: Any) -> ControllerPayload | None:
        """No payload published (``None``) unless scripted."""
        self._record("read_payload")
        scripted, answer = self._scripted("read_payload")
        return answer if scripted else None  # type: ignore[no-any-return]


#: Each Protocol and the mixin that offers it, in the order the composed class lists them.
MIXINS: dict[type, type] = {
    ArmTransport: _ArmTransport,
    ControllerKinematics: _ControllerKinematics,
    ControllerMotion: _ControllerMotion,
    TakesPayload: _TakesPayload,
    ReportsControllerModel: _ReportsControllerModel,
    HoldsAHaltLatch: _HoldsAHaltLatch,
    ControllerIO: _ControllerIO,
    ReadsForceTorque: _ReadsForceTorque,
    ReportsControllerStatus: _ReportsControllerStatus,
    ReleasesProtectiveStop: _ReleasesProtectiveStop,
    ReadsControllerProvenance: _ReadsControllerProvenance,
    ControllerFreedrive: _ControllerFreedrive,
    WatchesController: _WatchesController,
}

_CLASSES: dict[tuple[type, ...], type] = {}


def fake_transport(*, offers: Sequence[type] = tuple(MIXINS), **state: Any) -> Any:
    """A fake offering exactly the Protocols in ``offers`` (all twelve by default), built with ``state``."""
    unknown = [p for p in offers if p not in MIXINS]
    if unknown:
        raise ValueError(f"no fake offers {[getattr(p, '__name__', p) for p in unknown]}")
    if ArmTransport not in offers:
        raise ValueError("every transport offers ArmTransport: a planned arm cannot run without it")
    chosen = tuple(p for p in MIXINS if p in offers)
    cls = _CLASSES.get(chosen)
    if cls is None:
        name = "FakeTransport[" + "+".join(p.__name__ for p in chosen) + "]"
        cls = type(name, (*(MIXINS[p] for p in chosen), _State), {"__module__": __name__})
        from tests._injection_guard import guard

        guard(cls)
        _CLASSES[chosen] = cls
    return cls(**state)
