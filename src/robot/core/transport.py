"""What a planned arm needs from the controller below it (P0e E7).

``PlannedArm`` holds the judging and planning half every planned arm shares: the safety preflight, the cuRobo route,
the tool-frame composition, the camera world. What differs per vendor is how a command reaches a controller and how a
reading comes back, and that is a transport: one small required Protocol and a few optional ones in the style of
:mod:`~src.robot.core.arm_capabilities`, feature-checked with ``isinstance``.

Units and frames are the stack's. Joints are radians in the controller's own order (the arm description's joint
order). Every pose is a base-frame :class:`~src.geometry.Pose` in millimetres and XYZW, in the controller's
FRAME: the bare flange where the driver composes the tool (``tool_frame.source: willy``), the declared TCP where the
controller holds it. The composition itself stays with the arm, so a transport never learns what a tool is. A
vendor's own pose type (the UR ``URPose``) never crosses this boundary.

The verbs are what the UR driver already sends, named for what they do, so the UR transport is a translation and the
arm's commands do not change: ``tests/test_what_a_test_injects_reaches_the_arm.py`` and the P0e command tape hold that.

One protocol is Willy's own: :class:`HoldsAHaltLatch`, the halt latch the console's "halt now" sets (``SupportsHalt``).
It lives on the link, so a disconnect and a connect keep it, and a planned arm reads it before every leg it sends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Sequence

    import numpy as np

    from src.geometry import Pose

    from .arm_capabilities import ControllerProvenance, DigitalIOPort, HaltState, MoveEnd, RobotStatus, Wrench
    from .freedrive import ControllerPayload
    from .motion_result import MotionStatus

__all__ = [
    "ArmTransport",
    "ControllerFreedrive",
    "ControllerIO",
    "ControllerKinematics",
    "ControllerMotion",
    "HoldsAHaltLatch",
    "PlannedArmTransport",
    "ReadsControllerProvenance",
    "ReadsForceTorque",
    "ReleasesProtectiveStop",
    "ReportsControllerModel",
    "ReportsControllerStatus",
    "TakesPayload",
    "WatchesController",
]


@runtime_checkable
class ArmTransport(Protocol):
    """The verbs every planned arm's transport has.

    ``run_trajectory`` sends a joint path waypoint by waypoint and answers ``False`` at the first waypoint the
    controller rejects, leaving the rest unsent; what the controller raises (a dropped link) propagates, so the arm
    can tell a refusal from a fault. A UR whose control script stopped answers ``False`` here rather than raising
    (measured on URSim, 2026-09-22), which is why the steady gate asks first: ``wait_until_steady`` raises
    ``RobotConnectionError`` where the controller cannot say whether the arm is at rest, and its ``False`` means only
    that the arm is not at rest yet. ``vel`` and ``acc`` are passed through as given: limiting them is the arm's job,
    not the wire's. A ``ValueError`` from ``run_trajectory`` means the controller's library refused a value of the
    waypoint before sending it, a speed or an acceleration outside its range, so nothing of that waypoint was sent;
    a transport that has sent part of a waypoint when it fails does not raise ``ValueError`` (P0g R3).
    """

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_joints(self) -> list[float]: ...

    def run_trajectory(
        self, waypoints: Sequence[Sequence[float]], *, vel: float | None, acc: float | None,
    ) -> bool: ...

    def stop(self) -> None: ...

    def wait_until_steady(self, timeout_s: float, poll_interval_s: float) -> bool: ...


@runtime_checkable
class ControllerKinematics(Protocol):
    """A controller that solves its own kinematics and reports its own TCP register.

    ``controller_tcp_now`` is the register at this instant as a 4x4 in millimetres, which the tool-frame check pairs
    with the joints read beside it; it is a matrix rather than a pose because that check stores what it derives, and a
    round trip through a quaternion would change the bits the planner is handed. ``controller_ik`` answers ``None``
    where the controller finds no solution.
    """

    def is_steady(self) -> bool: ...

    def read_tcp(self, *, label: str = "") -> Pose: ...

    def controller_tcp_now(self) -> np.ndarray: ...

    def controller_fk(self, joints: Sequence[float], *, label: str = "") -> Pose: ...

    def controller_ik(self, pose: Pose, *, seed: Sequence[float] | None = None) -> list[float] | None: ...


@runtime_checkable
class ControllerMotion(Protocol):
    """A controller that moves to a Cartesian pose itself, after its own IK: the UR ik route.

    ``workspace_pose`` is what the controller-side workspace box judges when it differs from ``pose`` (in willy mode
    the controller is told the flange and the box bounds the grasp centre). ``last_reject_status`` is why the last
    ``move_pose`` answered ``False``, or ``None``.
    """

    def move_pose(
        self, pose: Pose, *, linear: bool, vel: float | None, acc: float | None, register: bool,
        workspace_pose: Pose,
    ) -> bool: ...

    @property
    def last_reject_status(self) -> MotionStatus | None: ...


@runtime_checkable
class TakesPayload(Protocol):
    """A controller told the carried mass and its centre of gravity (mm, flange frame) for its own stop model."""

    def set_payload(self, mass_kg: float, cog_mm: Sequence[float]) -> None: ...


@runtime_checkable
class ReportsControllerModel(Protocol):
    """A controller that says which arm it is, as its own firmware spells it, or ``None`` where it cannot."""

    def controller_model(self) -> str | None: ...


@runtime_checkable
class HoldsAHaltLatch(Protocol):
    """A link that keeps the halt latch (Willy's own): set, it sends no motion and switches no output until it is cleared.

    ``request_halt`` latches the link and sends nothing from the calling thread; a move in flight is braked by the
    thread that sent it where ``brake_on_halt`` says so (every UR), and runs to its end otherwise. ``halt_state`` is the latch
    while it is set. ``last_move_end`` says how the last move the link was asked for ended, read by the thread that
    asked for it, and ``halt_requests`` counts the latches set since the link was built, so a path that met a halt
    cleared again before its next leg still ends there. Both are ``None`` on a link that keeps no such record.
    """

    def request_halt(self, reason: str) -> HaltState: ...

    def clear_halt(self) -> None: ...

    def halt_state(self) -> HaltState | None: ...

    @property
    def brake_on_halt(self) -> bool: ...

    @property
    def last_move_end(self) -> MoveEnd | None: ...

    @property
    def halt_requests(self) -> int | None: ...


@runtime_checkable
class ControllerIO(Protocol):
    """A controller's digital and analog I/O registers, by bank (P0g slice 6).

    A write answers whether the controller took it: ``False`` means it was not taken, so the output may not stand
    where it was asked to (P0g io1), and what the link raises propagates. A read answers the register's level and
    raises where the controller cannot answer; it never guesses one. ``port`` is always named, because the bank is
    part of a pin's address. The verbs are named apart from ``SupportsDigitalIO``'s on purpose: a transport that
    matched the arm capability by name would pass for an I/O arm, and a gripper handed one would drop every
    ``False`` a write answers. ``end_pulse`` (Willy's own) drives a pin low past the halt latch, the one write a halted
    link makes, to end a pulse a halt cut short; on a link that keeps no latch it is the plain write of low.
    """

    def write_digital(self, pin: int, value: bool, *, port: DigitalIOPort) -> bool: ...

    def read_digital_input(self, pin: int, *, port: DigitalIOPort) -> bool: ...

    def read_digital_output(self, pin: int, *, port: DigitalIOPort) -> bool: ...

    def write_analog(self, pin: int, value: float, *, current: bool) -> bool: ...

    def end_pulse(self, pin: int, *, port: DigitalIOPort) -> bool: ...


@runtime_checkable
class ReadsForceTorque(Protocol):
    """A controller that reads the generalized force at the TCP and the torque at each joint (P0g slice 7).

    ``read_tcp_wrench`` is newtons and newton-metres in the base frame; ``read_joint_torques`` newton-metres in the
    controller's joint order. What the link raises propagates. Readings only: commanding a force is a later phase.
    """

    def read_tcp_wrench(self) -> Wrench: ...

    def read_joint_torques(self) -> tuple[float, ...]: ...


@runtime_checkable
class ReportsControllerStatus(Protocol):
    """A controller that reports its robot mode, safety mode, stop flags and its own text (P0g slice 7).

    As the core's :class:`~src.robot.core.arm_capabilities.RobotStatus`: what the vendor's integers mean is the
    transport's to know, and a mode it cannot name is ``UNKNOWN``. What the link raises propagates. The halt latch is
    not the controller's, so ``halted`` stays empty here; the arm reads it beside. ``read_status_quickly`` (Willy's
    own) answers from what the link already holds, with no round trip for the controller's text (``message`` empty),
    for a panel that polls.
    """

    def read_status(self) -> RobotStatus: ...

    def read_status_quickly(self) -> RobotStatus: ...


@runtime_checkable
class ReleasesProtectiveStop(Protocol):
    """A controller whose protective stop a person may release from software (P0g slice 7).

    ``True`` where the controller took the release. Which faults a person may clear this way is the arm's to decide; a
    transport only sends the release it was asked for, and nothing in the stack sends it on its own.
    """

    def release_protective_stop(self) -> bool: ...


@runtime_checkable
class WatchesController(Protocol):
    """A controller a program can watch without driving it (P0i, owner 2026-09-26).

    ``connect_watch`` opens what reads the controller's state (``ReportsControllerStatus``) and releases a protective
    stop (``ReleasesProtectiveStop``), and never starts a control program, so it opens on a controller in a protective
    stop, where a control program will not start: measured on URSim, 2026-09-26, where a full connect was refused and
    the fault door with it. ``is_watching`` is true while that link is open; ``is_connected`` stays false on it, so no
    verb that needs a control program is sent over it, and ``disconnect`` closes it as it closes a connection. On a
    watch no control program runs, so the status says ``program_running=False`` by design. A controller whose status
    or release needs its control program must not offer this.
    """

    @property
    def is_watching(self) -> bool: ...

    def connect_watch(self) -> None: ...


@runtime_checkable
class ReadsControllerProvenance(Protocol):
    """A controller that says what it is: its serial, the address it is reached at, and whether that proves a
    simulator (P0g slice 7).

    The rule that turns the evidence into proof is the vendor's, so it lives in the transport; what the link raises
    propagates, and the arm, whose capability never raises, turns it into a provenance that says nothing.
    """

    def read_provenance(self) -> ControllerProvenance: ...


@runtime_checkable
class ControllerFreedrive(Protocol):
    """A controller that lets a person move the arm by hand while the program watches.

    Offered only by a controller that holds the arm on its own when no kick arrives within ``1 / min_frequency_hz``
    of an armed watchdog: a controller that cannot must not offer it, because an arm freed by a program that then
    died would stay free in a person's hands. The UR's RTDE watchdog protective-stops the control program, and with
    it teach mode. ``arm_watchdog``, ``kick_watchdog``, ``begin_hand_guiding`` and ``end_hand_guiding`` answer whether
    the controller took the command; ``restart_control_program`` starts a fresh control program with no watchdog and
    no hand guiding, and raises where none comes up. ``control_program_running`` and ``stop_latched`` are the cheap
    reads a 30 Hz loop may make; ``read_status`` is not one. ``read_joint_speeds`` answers radians per second in the
    controller's joint order, and ``read_payload`` the payload the controller compensates for, or ``None`` where it
    does not publish one. The verbs are named apart from ``SupportsFreedrive``'s, as ``ControllerIO``'s are.
    """

    def arm_watchdog(self, min_frequency_hz: float) -> bool: ...

    def kick_watchdog(self) -> bool: ...

    def begin_hand_guiding(self) -> bool: ...

    def end_hand_guiding(self) -> bool: ...

    def restart_control_program(self) -> None: ...

    def control_program_running(self) -> bool: ...

    def stop_latched(self) -> bool: ...

    def read_joint_speeds(self) -> list[float]: ...

    def read_payload(self) -> ControllerPayload | None: ...


@runtime_checkable
class PlannedArmTransport(
    ArmTransport, ControllerKinematics, ControllerMotion, TakesPayload, ReportsControllerModel, HoldsAHaltLatch,
    Protocol,
):
    """Every verb ``PlannedArm`` sends today, which is the whole surface of the UR transport (P0e E8).

    The planned arm was cut out of the UR driver without changing what a method does, so it still asks for the
    controller's own kinematics, its Cartesian moves, its payload register and its halt latch on the paths that use
    them. The slices after the move narrow what it needs to :class:`ArmTransport` and check the optional ones with
    ``isinstance``, which is what a transport without a controller IK will offer.
    """
