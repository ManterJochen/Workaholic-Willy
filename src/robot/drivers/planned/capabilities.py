"""The optional capabilities a planned arm offers over its transport, one mixin each (P0g slices 6 and 7).

A capability Protocol of :mod:`~src.robot.core.arm_capabilities` is checked structurally: a gripper, the
build's verdict and the desk each ask ``isinstance(arm, SupportsDigitalIO)``. So a capability cannot be a method of
``PlannedArm``, where every planned arm would claim it; it is a mixin a vendor class lists when its transport offers
the matching transport Protocol of :mod:`~src.robot.core.transport`, and an arm whose transport has no I/O
lists none and is refused a vacuum or an I/O jaw by type. A mixin never subclasses the Protocol it implements: a
Protocol base would lend its stubs to the class and make the check pass on nothing.

Each mixin translates only what the transport cannot say in the arm's terms: a write the controller did not take is a
``RobotConnectionError`` here, and a ``False`` on the transport.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol, cast

from src.robot.core import (
    ArmFaultClass,
    ControllerPayload,
    ControllerProvenance,
    FreedriveSession,
    RobotConnectionError,
    RobotMotionRejected,
    RobotStatus,
    Wrench,
)
from src.robot.core.arm_capabilities import DigitalIOPort, halted_refusal
from src.robot.core.freedrive import HAND_GUIDED_MESSAGE

if TYPE_CHECKING:  # pragma: no cover (typing only)
    import logging
    from collections.abc import Callable

    from src.robot.core import ArmFault, HaltState, RobotArm
    from src.robot.core.transport import (
        ArmTransport,
        ControllerFreedrive,
        ControllerIO,
        ReadsControllerProvenance,
        ReadsForceTorque,
        ReleasesProtectiveStop,
        ReportsControllerStatus,
        WatchesController,
    )

    class _FreedriveTransport(ArmTransport, ControllerFreedrive, ReportsControllerStatus, Protocol):
        """What :class:`PlannedFreedrive` needs of the transport, for the checker only."""

__all__ = [
    "PlannedControllerProvenance",
    "PlannedDigitalIO",
    "PlannedFaultAcknowledge",
    "PlannedForceTorque",
    "PlannedFreedrive",
    "PlannedRobotStatus",
    "PlannedWatching",
]


def _not_taken(port: object, pin: int, value: bool) -> RobotConnectionError:
    """The refusal of a digital write the controller reported undelivered, naming the bank, the pin and the level."""
    level = "high" if value else "low"
    return RobotConnectionError(
        f"the controller did not take {port} digital output {int(pin)} = {level}: ur_rtde reported the write "
        f"undelivered, so the output may not stand where it was asked to (P0g io1)."
    )


class PlannedDigitalIO:
    """``SupportsDigitalIO`` over a ``ControllerIO`` transport (P0g slice 6, io2).

    A write the controller did not take raises ``RobotConnectionError``, so no caller moves its belief on a write that
    never happened (P0g io1); a read answers the register as the transport does.
    """

    _transport: ControllerIO

    def set_digital_output(self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        """Drive a controller digital output pin (bank ``port``) high/low; a write the controller did not take raises."""
        if not self._transport.write_digital(int(pin), bool(value), port=port):
            raise _not_taken(port, pin, value)

    def end_output_pulse(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        """Drive ``pin`` low to end a pulse that began before a halt: the one output write a halted arm makes.

        For a bistable valve's coil and a vacuum blow-off, whose pulse must end low however it ends; never for a toggle
        hand, whose every change of its output moves the jaws (``ControllerIO.end_pulse``). A write the controller did
        not take raises.
        """
        if not self._transport.end_pulse(int(pin), port=port):
            raise _not_taken(port, pin, False)

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read a controller digital input pin (bank ``port``)."""
        return self._transport.read_digital_input(int(pin), port=port)

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read back a controller digital output pin's commanded level (bank ``port``)."""
        return self._transport.read_digital_output(int(pin), port=port)

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        """Set a controller analog output: voltage (V) unless ``current=True`` (A); a write not taken raises."""
        if not self._transport.write_analog(int(pin), float(value), current=bool(current)):
            unit = "A" if current else "V"
            raise RobotConnectionError(
                f"the controller did not take analog output {int(pin)} = {float(value):g} {unit}: ur_rtde reported the "
                f"write undelivered (P0g io1)."
            )


class PlannedForceTorque:
    """``SupportsForceTorque`` over a ``ReadsForceTorque`` transport (P0g slice 7): the hand-over signals."""

    _transport: ReadsForceTorque

    def get_tcp_wrench(self) -> Wrench:
        """Live generalized force/torque at the TCP (N, N·m, BASE frame) -- the hand-over signal."""
        return self._transport.read_tcp_wrench()

    def get_joint_torques(self) -> tuple[float, ...]:
        """Live torque at each joint (newton-metres), base-to-tool order."""
        return self._transport.read_joint_torques()


class PlannedRobotStatus:
    """``get_robot_status`` and ``quick_robot_status`` over a ``ReportsControllerStatus`` transport (P0g slice 7).

    Each carries the halt latch, read beside the controller (``halted``), so ``is_operational`` refuses a halted arm
    and ``controller_operational`` keeps the controller's own answer. ``SupportsRobotStatus`` is this and
    :class:`PlannedFaultAcknowledge` together, so a vendor lists both.
    """

    _transport: ReportsControllerStatus
    _halt_record: Callable[[], HaltState | None]

    def get_robot_status(self) -> RobotStatus:
        """A snapshot of the controller's live robot mode, safety mode, stop flags and text, and the halt latch.

        The enriched controller state a pipeline surfaces instead of a bare motion rejection, in the core's terms; what
        the vendor's integers mean is the transport's.
        """
        halted = self._halt_record()
        return _with_the_latch(halted, self._transport.read_status())

    def quick_robot_status(self) -> RobotStatus:
        """The same with no round trip for the controller's text (``message`` empty): what a ready bar polls."""
        halted = self._halt_record()
        return _with_the_latch(halted, self._transport.read_status_quickly())


def _with_the_latch(halted: HaltState | None, status: RobotStatus) -> RobotStatus:
    """``status`` with the halt latch's reason in ``halted``, where it is set."""
    return dataclasses.replace(status, halted=halted.reason) if halted is not None else status


class PlannedFaultAcknowledge:
    """``acknowledge_fault`` over a ``ReleasesProtectiveStop`` transport (P0g slice 7)."""

    _transport: ReleasesProtectiveStop
    logger: logging.Logger

    def acknowledge_fault(self, fault: ArmFault) -> bool:
        """A PERSON acknowledges ``fault`` at this controller; ``True`` when it acknowledged.

        Recovery is never automatic (owner, 2026-09-19): nothing in the pick path, the grasping stack or the
        safety layer may call this, and a boundary test says so. It is reached from the CLI, where the person
        who clears a stop is standing at the cell and can see the arm.

        Only a protective stop is clearable from software here. An emergency button is released by hand and a
        powered-off controller is powered on by hand, so this answers ``False`` and says which it was rather
        than sending a release that cannot work.
        """
        if fault.kind is not ArmFaultClass.PROTECTIVE_STOP:
            self.logger.warning(
                "a %s is not acknowledged from software: %s", fault.kind.value, fault.what_to_do())
            return False
        ok = self._transport.release_protective_stop()
        if ok:
            self.logger.info("Requested protective-stop release from the UR dashboard.")
        else:
            self.logger.warning("Protective-stop release unavailable (no dashboard connection).")
        return ok


class PlannedWatching:
    """``SupportsWatching`` over a ``WatchesController`` transport (P0i, owner 2026-09-26).

    The fault door's link: it reads the status and takes a person's acknowledgement on a controller in a protective
    stop, where a full connect is refused (measured on URSim, 2026-09-26). The transport is not connected on it, so
    every motion verb of the arm refuses with nothing sent, as it does on an arm not connected; ``disconnect`` closes
    it. A vendor lists this beside ``PlannedRobotStatus`` and ``PlannedFaultAcknowledge`` only where its transport's
    status and release need no control program.
    """

    _transport: WatchesController

    @property
    def is_watching(self) -> bool:
        return bool(self._transport.is_watching)

    def watch(self) -> None:
        """Open the watch; a no-op on an arm that is connected or already watching."""
        self._transport.connect_watch()


class PlannedControllerProvenance:
    """``ReportsControllerProvenance`` over a ``ReadsControllerProvenance`` transport (P0g slice 7).

    Best effort and never raising: a controller with no dashboard, or one that will not answer this tick, says nothing
    at all, and nothing is a different answer from "it is real" (see :class:`ControllerProvenance`).
    """

    _transport: ReadsControllerProvenance
    logger: logging.Logger

    def controller_provenance(self) -> ControllerProvenance:
        """What this arm's controller says about itself, as its transport reads it, or nothing where it cannot."""
        try:
            return self._transport.read_provenance()
        except Exception as exc:  # noqa: BLE001 (evidence gathering never breaks a status panel)
            self.logger.debug("no controller provenance this tick: %s: %s", type(exc).__name__, exc)
            return ControllerProvenance()


class PlannedFreedrive:
    """``SupportsFreedrive`` over a transport that offers ``ControllerFreedrive``, ``ReportsControllerStatus`` and
    ``ArmTransport`` (port S4).

    ``freedrive()`` hands out a :class:`~.freedrive.PlannedFreedriveSession`, and from its with-block's start to its end
    every motion verb of the arm refuses (``PlannedArm._hand_guided_refusal``, which reads the session this records). A
    vendor lists this mixin only where its controller holds the arm when the watchdog's kicks stop, which is the
    transport Protocol's contract.
    """

    _transport: _FreedriveTransport
    _freedrive: FreedriveSession | None
    _halt_record: Callable[[], HaltState | None]
    _forget_what_was_judged_ahead: Callable[[], None]
    logger: logging.Logger

    def freedrive(self) -> FreedriveSession:
        """A hand-guiding session; nothing is freed yet.

        Refused while the arm is not connected, while a session is open, while the arm is halted (in the halt's own
        words) and while the controller is not operational (a stop, a safety mode other than normal, brakes engaged,
        no control program). Whatever was judged ahead for the next move is forgotten: a person is about to move the
        arm. The session frees the arm on ``free()`` inside its with-block; :mod:`.freedrive` has the order of the
        controller calls.
        """
        from .freedrive import PlannedFreedriveSession, raise_unless_operational

        self._forget_what_was_judged_ahead()
        if not self._transport.is_connected:
            raise RobotConnectionError("freedrive() requires an open connection.")
        if self._freedrive is not None:
            raise RobotMotionRejected(f"{HAND_GUIDED_MESSAGE}; leave it before opening another")
        halted = self._halt_record()
        if halted is not None:
            # Said as the halt: the controller sentence a halted status reads in raise_unless_operational would name a
            # controller that is fine.
            raise RobotMotionRejected(halted_refusal(halted.reason, "a freedrive session was not opened"))
        raise_unless_operational(self._status_with_the_latch(), "a freedrive session")
        return PlannedFreedriveSession(
            self._transport, tcp_pose=cast("RobotArm", self).get_tcp_pose, robot_status=self._status_with_the_latch,
            on_open=self._freedrive_opened, on_close=self._freedrive_closed, logger=self.logger,
        )

    def controller_payload(self) -> ControllerPayload | None:
        """The payload the controller compensates for, or ``None`` where it does not publish one.

        It decides whether a freed arm floats, sinks or rises, and it is the controller's own record, which is the one
        ``safety.payload`` pushed at connect only where that says ``enforce: true``.
        """
        return self._transport.read_payload()

    def _status_with_the_latch(self) -> RobotStatus:
        """The controller's state with the halt latch beside it, as ``PlannedRobotStatus`` reads it."""
        return _with_the_latch(self._halt_record(), self._transport.read_status())

    def _freedrive_opened(self, session: FreedriveSession) -> None:
        """Record ``session`` as the open one; another open session is refused."""
        if self._freedrive is not None and self._freedrive is not session:
            raise RobotMotionRejected(f"{HAND_GUIDED_MESSAGE}; leave it before opening another")
        self._freedrive = session

    def _freedrive_closed(self, session: FreedriveSession) -> None:
        """Forget ``session``, which gives motion back."""
        if self._freedrive is session:
            self._freedrive = None
