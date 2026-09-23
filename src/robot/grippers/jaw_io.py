"""Parallel-jaw gripper driven through the robot controller's digital I/O.

This is the vendor-neutral twin of :mod:`.vacuum`, on the same rule: a pneumatic or
electric two-finger gripper on a UR is one or two output pins plus, usually, reed
switches on inputs. There is no SDK to import and nothing manufacturer-specific in
that loop, so a cell can be configured for a jaw gripper before anyone knows which
one will be bought, and bring-up becomes measuring pin numbers rather than writing
code.

It is distinct from :mod:`.robotiq`, which is also jaws: the Robotiq speaks a socket
protocol on port 63352 that the URCap opens, not I/O. Both can be mounted on the same
cell, and only the ``vendor`` string changes.

Width semantics. The :class:`Gripper` Protocol is width-based because jaws travel. A
digital-I/O jaw does not travel: it is open or closed and nothing in between is
commandable. Width is therefore reinterpreted exactly as the suction driver
reinterprets it, which keeps the two I/O end-effectors honest about each other::

    set_width_mm(w)   w <= closed_below_mm  ->  close
                      w >  closed_below_mm  ->  open

``speed`` and ``force`` are accepted for Protocol parity and ignored, because a
solenoid has one setting. Regulating either means a pressure regulator or a drive,
neither of which is on a digital pin.

The payoff is feedback, and specifically two reed switches. With a fully-open and a
fully-closed switch the driver tells apart the two outcomes a jaw cell otherwise
cannot distinguish:

===================  ==================  =========================================================
``open_confirm``     ``closed_confirm``  meaning
===================  ==================  =========================================================
True                 False               jaws fully open
False                True                fully closed, so the jaws met and the grasp is empty
False                False               stopped in between, so something is between them
True                 True                contradictory wiring or sensor; fail closed, no object
===================  ==================  =========================================================

That makes this driver an :class:`ObjectDetectingGripper`, which opts the cell into
post-close verification in ``GraspExecutionPolicy``. The suction path is otherwise
the only one that can be asked whether it is holding something, because the Robotiq
driver exposes no detection capability and ``WidthDeltaGripperVerifier`` needs a
travelling jaw to measure.

Status is bucket 3: this has never touched a real gripper. The I/O calls it makes are
the ones the UR driver already exposes. What is unverified is the wiring, meaning the
pin numbers, which port block, whether the reed switches are active-high, whether the
valve is single-acting, double-acting or a toggle that flips on every pulse, and how
long the cylinder takes to travel on the actual hardware. Every one of those is a
field on :class:`~src.config.schema.robot.robot_schema.JawIOGripperConfig`, so
bring-up is a matter of measuring numbers rather than editing this file.
"""

from __future__ import annotations

import os
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.robot.core import RobotConnectionError, RobotError
from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO
from src.robot.core.gripper import HoldEvidence

from ..constants import JAW_IO_GRIPPER_LOG_FILE, create_robot_logger
from . import jaw_toggle_state

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot.robot_schema import GripperConfig

__all__ = ["JawIOGripper", "JawState"]


class JawState(StrEnum):
    """What the wired feedback says the jaws are doing.

    ``UNKNOWN`` is a first-class answer rather than a failure. A cell with no feedback
    pins, or with only a fully-open switch, which cannot tell closed on a part from
    closed on nothing, has genuinely not measured anything. Reporting that honestly is
    the point, because the alternative is a driver that invents a verdict and a
    verification stage that trusts it.
    """

    OPEN = "open"
    #: Fully closed after a close command: the jaws met, so nothing was between them.
    CLOSED_EMPTY = "closed_empty"
    #: Stopped between the end stops: something is between the jaws.
    HOLDING = "holding"
    #: Both end-stop switches active at once. That is mechanically impossible, so it is
    #: the wiring or a failed sensor. Treated as no object, fail closed, and never as a
    #: successful grasp.
    CONTRADICTORY = "contradictory"
    UNKNOWN = "unknown"


class JawIOGripper:
    """A parallel-jaw gripper actuated over the controller's digital I/O.

    ``io`` is any object satisfying :class:`SupportsDigitalIO`. In a real cell that is
    the arm driver itself, because the valve is wired to the controller. The gripper
    never opens a connection of its own: it borrows the arm's, so ``connect()`` and
    ``disconnect()`` reflect whether the I/O source is usable rather than owning a
    socket. The lifecycle consequence is the same as for suction: the arm must be
    connected before the gripper is.
    """

    def __init__(
        self,
        io: SupportsDigitalIO,
        *,
        config: "GripperConfig | None" = None,
        actuation: str = "single_solenoid",
        close_output_pin: int = 0,
        open_output_pin: int | None = None,
        pulse_s: float = 0.2,
        part_present_input_pin: int | None = None,
        closed_confirm_input_pin: int | None = None,
        open_confirm_input_pin: int | None = None,
        io_port: DigitalIOPort | str = DigitalIOPort.TOOL,
        close_timeout_s: float = 1.0,
        close_settle_s: float = 0.3,
        closed_below_mm: float = 5.0,
        open_on_connect_without_feedback: bool = False,
        min_width_mm: float = 0.0,
        max_width_mm: float = 85.0,
        state_file: "str | Path | None" = None,
        sleep: Any = time.sleep,
    ) -> None:
        if not isinstance(io, SupportsDigitalIO):
            raise TypeError(
                "JawIOGripper needs a digital-I/O source (the arm driver on a real cell): "
                f"{type(io).__name__} does not satisfy SupportsDigitalIO."
            )
        if actuation not in ("single_solenoid", "double_solenoid", "single_toggle"):
            raise ValueError(
                f"JawIOGripper: unknown actuation {actuation!r}; expected 'single_solenoid' "
                "(one output, spring return), 'double_solenoid' (two pulsed outputs, bistable) "
                "or 'single_toggle' (one pulsed output, each pulse flips the jaws)."
            )
        if actuation == "double_solenoid" and open_output_pin is None:
            # The same refusal the config schema makes, repeated here because the
            # driver is constructible directly, and a bistable valve with no open coil
            # would latch closed on the first grasp and never release.
            raise ValueError(
                "JawIOGripper: actuation 'double_solenoid' needs open_output_pin; a bistable valve "
                "has no spring to open it."
            )
        if actuation == "single_solenoid" and open_output_pin is not None:
            # A single solenoid opens by dropping the close pin, so a second pin would
            # never be driven. Accepting it silently is how a cell ends up with a wire
            # nobody notices is dead, so this refuses as the schema does, because the
            # constructor is reachable without the schema.
            raise ValueError(
                "JawIOGripper: actuation 'single_solenoid' opens by dropping close_output_pin, so "
                "open_output_pin is never driven. Drop it, or use 'double_solenoid'."
            )
        if actuation == "single_toggle" and open_output_pin is not None:
            # The schema's refusal, repeated for the same reason as the one above: a toggle
            # opens by pulsing close_output_pin again, so a second pin is a wire nobody drives.
            raise ValueError(
                "JawIOGripper: actuation 'single_toggle' opens by a second pulse on "
                "close_output_pin, so open_output_pin is never driven. Drop it, or use "
                "'double_solenoid'."
            )
        if actuation == "single_toggle" and open_on_connect_without_feedback:
            # A toggle can only flip the jaws, and a pulse on jaws that already stand open
            # closes them. This refuses as the schema does, because the constructor is
            # reachable without the schema.
            raise ValueError(
                "JawIOGripper: actuation 'single_toggle' cannot assert 'open' on connect: every "
                "pulse flips the jaws, so one on jaws already open would close them. Drop "
                "open_on_connect_without_feedback."
            )
        if (actuation == "single_toggle" and closed_confirm_input_pin is not None
                and open_confirm_input_pin is None):
            # Off its stop a lone closed switch can only repeat what was last commanded
            # (_read_state), so a lost pulse would be read back as a grasp and the next
            # pulse would flip the jaws the wrong way.
            raise ValueError(
                "JawIOGripper: actuation 'single_toggle' with closed_confirm_input_pin and no "
                "open_confirm_input_pin: off the closed stop that switch only repeats what was "
                "last commanded, so a lost pulse would be read back as a grasp. Wire "
                "open_confirm_input_pin too, or neither."
            )
        if open_output_pin is not None and int(open_output_pin) == int(close_output_pin):
            raise ValueError(
                f"JawIOGripper: close_output_pin and open_output_pin are both {close_output_pin}; "
                "one pin cannot drive both coils."
            )
        self._io = io
        self._actuation = actuation
        self._close_pin = int(close_output_pin)
        self._open_pin = None if open_output_pin is None else int(open_output_pin)
        self._pulse_s = float(pulse_s)
        self._part_pin = None if part_present_input_pin is None else int(part_present_input_pin)
        self._closed_pin = (
            None if closed_confirm_input_pin is None else int(closed_confirm_input_pin)
        )
        self._open_confirm_pin = (
            None if open_confirm_input_pin is None else int(open_confirm_input_pin)
        )
        self._port = DigitalIOPort(io_port) if not isinstance(io_port, DigitalIOPort) else io_port
        self._close_timeout_s = float(close_timeout_s)
        self._close_settle_s = float(close_settle_s)
        self._closed_below_mm = float(closed_below_mm)
        self._open_on_connect_without_feedback = bool(open_on_connect_without_feedback)
        if config is not None:
            min_width_mm, max_width_mm = float(config.min_width_mm), float(config.max_width_mm)
        self._min_width_mm = float(min_width_mm)
        self._max_width_mm = float(max_width_mm)
        self._sleep = sleep
        self._connected = False
        # What the driver last commanded, not what the jaws are doing. It is kept apart
        # from the sensed state on purpose: conflating the two is how a gripper reports
        # a grasp it never made. A single_toggle has no level to command, so for it this
        # is the driver's belief about where the jaws stand: moved at each pulse's edge,
        # reset at every connect, and set from the open switch where one is wired and
        # measured otherwise.
        self._closed = False
        # A toggle pulse whose high write failed, so whether the jaws flipped is
        # unknowable. A cell with no open switch refuses to pulse again until a connect
        # resets the belief.
        self._edge_unknown = False
        # A toggle's open pulse whose wait for the open stop did not finish, because an
        # exception came mid-stroke. The next decision waits the stroke out first, or it
        # would read travelling jaws as closed and pulse them back.
        self._awaiting_open = False
        # Where a toggle's count of its own pulses is kept between programs
        # (jaw_toggle_state). None keeps it in memory only, as before, which a driver
        # built directly gets; the cell's builder names a file per controller, bank and pin.
        self._state_file = (Path(state_file) if state_file is not None and actuation == "single_toggle"
                            else None)
        # The record says a pulse was started and not seen to finish: nobody knows where
        # the jaws stand until a person has looked and declared it (declare_jaws).
        self._record_in_flight = False
        self.logger = create_robot_logger("JawIOGripper", JAW_IO_GRIPPER_LOG_FILE)

    # --- state ------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def min_width_mm(self) -> float:
        return self._min_width_mm

    @property
    def max_width_mm(self) -> float:
        return self._max_width_mm

    @property
    def closed_below_mm(self) -> float:
        """The commanded width at or below which the jaws close; above it they open (``TwoStateGripper``)."""
        return self._closed_below_mm

    @property
    def has_feedback(self) -> bool:
        """Whether any feedback pin is wired. False means every verdict is a commanded state."""
        return any(
            pin is not None
            for pin in (self._part_pin, self._closed_pin, self._open_confirm_pin)
        )

    @property
    def jaws_closed(self) -> bool:
        """Whether the jaws are currently commanded closed, the driver's view and not the sensors.

        For a ``single_toggle``, which commands flips rather than levels, this is the
        driver's belief about where they stand (see ``_closed``).
        """
        return self._closed

    # --- lifecycle --------------------------------------------------------
    def connect(self) -> None:
        """Adopt the arm's I/O and reach a safe state without blindly dropping a workpiece.

        The rule differs from the suction driver's on purpose. Suction asserts off on
        connect, because releasing a cup that was left latched is cheap. A jaw gripper
        left closed from an interrupted run may be holding a rigid part, and opening it
        drops that part wherever the arm happens to be standing. So:

        * feedback says the jaws are empty, open, and the cell starts in a known state;
        * feedback says something is held, hold it and warn, so a human decides;
        * no feedback wired, do not actuate. Proven empty is unreachable without a
          sensor, and the point of the rule is not to drop what has not been checked.
          ``open_on_connect_without_feedback=True`` opts into the suction driver's
          behaviour instead.

        Nothing here commands motion in the held case, so a cell that boots holding a
        part stays exactly as the last run left it.

        A ``single_toggle`` connects by its own rule, :meth:`_connect_toggle`: every
        pulse flips its jaws, so it cannot assert a state.
        """
        self._connected = True
        if self._actuation == "single_toggle":
            self._connect_toggle()
            return
        state = self._read_state()
        if state is JawState.HOLDING:
            self.logger.warning(
                "connect(): the jaws report a part between them (%s). not opening; releasing here "
                "would drop it wherever the arm is standing. Clear it before running.", state.value,
            )
            self._closed = True
            return
        if state is JawState.CONTRADICTORY:
            self.logger.warning(
                "connect(): both end-stop switches read active, which is mechanically impossible; "
                "check the wiring / sensors. not actuating on an unreadable state.",
            )
            return
        if state is JawState.UNKNOWN and not self._open_on_connect_without_feedback:
            self.logger.info(
                "connect(): no feedback wired, so the jaws cannot be proven empty; not actuating. "
                "Set open_on_connect_without_feedback=true to assert an open state instead.",
            )
            return
        self._actuate(close=False)
        # Which pins, and what the jaws read before being opened. Those two facts make
        # a later report that it dropped the part on connect answerable, not arguable.
        self.logger.info(
            "connected on %s: close pin=%s, open pin=%s, feedback=%s; jaws read %s -> opened",
            self._port.value, self._close_pin,
            self._open_pin if self._open_pin is not None else "none",
            "wired" if self.has_feedback else "none", state.value,
        )

    def disconnect(self) -> None:
        """Detach without releasing. Never raises.

        This is deliberately not the mirror of
        :meth:`.vacuum.VacuumGripper.disconnect`, which releases on the way out.
        Dropping a held part during teardown, which is often an error path, is how a
        workpiece ends up on the floor of a cell nobody is watching. Releasing is an
        explicit ``set_width_mm`` call, so a caller that wants it asks for it.
        """
        self._connected = False
        # This says what was left behind, because the driver deliberately does not
        # release: a cell found holding a part next morning is explained by this line.
        left = ("UNKNOWN, the last pulse failed on its high write" if self._edge_unknown
                else "CLOSED" if self._closed else "open")
        if self._actuation == "single_toggle" and (self._closed or self._edge_unknown):
            # A toggle left closed is where the next program starts inverted, unless it reads
            # the record; said at warning level so the line is there when that program misbehaves.
            kept = (f"recorded in {self._state_file}, and the next program starts from it" if self._state_file
                    else "nothing records it, and the next program will take them to stand OPEN")
            self.logger.warning("disconnected without releasing (jaws left %s); %s", left, kept)
            return
        self.logger.info("disconnected without releasing (jaws left %s)", left)

    def activate(self) -> None:
        """No homing to run, because a solenoid jaw has no calibration. Kept for Protocol parity."""
        self._require_connected("activate")

    # --- commands ---------------------------------------------------------
    def set_width_mm(
        self, width_mm: float, *, speed: float | None = None, force: float | None = None,
    ) -> None:
        """Reinterpret width as jaws open or closed, closing at or below ``closed_below_mm``.

        ``speed`` and ``force`` are accepted for Protocol parity and ignored, because a
        solenoid has one setting. Closing waits for the jaws to reach a settled state,
        because commanding the valve and driving away immediately is how a cell drops
        parts: a cylinder needs travel time, and on this hardware that time is the only
        thing standing between commanded and gripped.
        """
        self._require_connected("set_width_mm")
        self.set_closed(float(width_mm) <= self._closed_below_mm)

    def set_closed(self, closed: bool) -> None:
        """Close or open the jaws by intent, whatever ``closed_below_mm`` says (``OpensAndCloses``).

        The width a caller sends is read against ``closed_below_mm``, and a grasp above it
        is an open (owner's cell, 2026-09-23). A verb that knows what it means, a pick's
        pre-open, its grasp, a release, says it here instead, so no width can turn it round.
        A close waits for the jaws exactly as a closing width does.
        """
        self._require_connected("set_closed")
        close = bool(closed)
        was_closed = self._closed
        self._actuate(close=close)
        if close:
            self._await_close()
            if self._actuation == "single_toggle":
                self._settle_toggle_close()
        elif was_closed and not self._closed and self._open_confirm_pin is None:
            # The jaws travel the same stroke open as closed, and with no open switch the travel time is the only
            # wait there is. Without it a place returned the moment the valve was told, and the arm backed out of a
            # Hand-E that takes up to about 2 s to open, dragging the part it had just set down (owner's cell,
            # 2026-09-23).
            self._sleep(self._close_settle_s)

    def declare_jaws(self, *, closed: bool) -> None:
        """A person looked at a toggle's jaws and says where they stand; the driver and its record take it.

        The way out after a pulse nobody counted (the pendant's I/O tab, a power cut mid
        stroke) and after a record that says a pulse was started and not seen to finish.
        Nothing is pulsed. Refused for a toggle with an open switch, which reads the jaws
        itself, and for the solenoids, which command a level rather than a flip.
        """
        if self._actuation != "single_toggle" or self._open_confirm_pin is not None:
            raise RobotError(
                "JawIOGripper.declare_jaws: only a single_toggle with no open switch keeps a count of "
                "its own pulses; this one reads or commands where its jaws stand."
            )
        self._closed = bool(closed)
        self._edge_unknown = False
        self._record_in_flight = False
        if self._state_file is not None:
            jaw_toggle_state.declare(self._state_file, closed=bool(closed))
        self.logger.warning("jaws declared %s by a person; nothing pulsed", "CLOSED" if closed else "OPEN")

    def get_width_mm(self) -> float:
        """Reported opening: the closed band while closed, the open band while open.

        It is binary by construction, because there is no encoder on a solenoid, so
        this reports the band the jaws were commanded to, exactly as the suction driver
        reports its two bands. A caller that needs a genuinely measured width needs a
        gripper with a position sensor rather than this one.
        """
        return self._min_width_mm if self._closed else self._max_width_mm

    def is_object_detected(self) -> bool:
        """``True`` when the wired feedback says a part is between the jaws.

        The precedence is deliberate. A dedicated part-present sensor answers this
        exact question directly, so it wins where one is wired. The reed pair answers
        it by inference, from the jaws stopping short of their end stop, which is
        strong but second-hand. With neither, this reports the commanded state, the
        honest answer as far as this driver knows, and the same posture the suction
        driver takes without a switch.
        """
        self._require_connected("is_object_detected")
        if not self._closed_for_evidence():
            # Open jaws can hold nothing, and the reed pair reads as between the end
            # stops during travel, which would otherwise look exactly like a grasp.
            return False
        if self._part_pin is not None:
            return bool(self._io.get_digital_input(self._part_pin, port=self._port))
        state = self._read_state()
        if state is JawState.UNKNOWN:
            return True  # the gate above found the jaws closed
        return state is JawState.HOLDING

    def hold_evidence(self) -> HoldEvidence:
        """The wired feedback as evidence, never the command.

        UNMEASURED while the jaws are commanded open, whatever a part pin reads, because open
        jaws hold nothing and the reeds read between the stops during travel; for a
        ``single_toggle`` with an open switch, while that switch reads open
        (``_closed_for_evidence``). Closed, a part pin that reads high is HELD and low is
        EMPTY. With no part pin, ``JawState.HOLDING`` is HELD, ``JawState.CLOSED_EMPTY`` is
        EMPTY, and any other reed state, or no reeds at all, is UNMEASURED.
        """
        self._require_connected("hold_evidence")
        if not self._closed_for_evidence():
            return HoldEvidence.UNMEASURED
        if self._part_pin is not None:
            held = bool(self._io.get_digital_input(self._part_pin, port=self._port))
            return HoldEvidence.HELD if held else HoldEvidence.EMPTY
        state = self._read_state()
        if state is JawState.HOLDING:
            return HoldEvidence.HELD
        if state is JawState.CLOSED_EMPTY:
            return HoldEvidence.EMPTY
        return HoldEvidence.UNMEASURED

    def _closed_for_evidence(self) -> bool:
        """Whether the jaws count as closed for the grasp evidence.

        This is what was commanded, except on a toggle with an open switch. A toggle's
        belief can lag its jaws, after a lost edge or a write that raised after reaching
        the controller, and a part-present sensor reads a part between open jaws too. So
        on a toggle the open switch, where wired, is read instead: open jaws hold nothing,
        whatever the belief says. The solenoids assert a level or a coil and keep the
        commanded state, as before.
        """
        if self._actuation == "single_toggle" and self._open_confirm_pin is not None:
            return not bool(self._io.get_digital_input(self._open_confirm_pin, port=self._port))
        return self._closed

    def read_jaw_state(self) -> JawState:
        """The sensed jaw state, for operators and bring-up. Not part of any Protocol."""
        self._require_connected("read_jaw_state")
        return self._read_state()

    # --- internals --------------------------------------------------------
    def _require_connected(self, what: str) -> None:
        if not self._connected:
            raise RobotConnectionError(f"JawIOGripper.{what} requires connect() first.")

    def _actuate(self, *, close: bool) -> None:
        """Drive the valve. A single solenoid holds a level; a double one pulses the right coil;
        a toggle pulses its one pin only when the jaws stand the other way from the request."""
        # Info, not debug: the robot logger keeps info, and whether a command drove the jaws at
        # all is the first thing a cell whose jaws did the wrong thing asks of its log (owner's
        # cell, 2026-09-23).
        self.logger.info("actuating jaws %s via %s", "CLOSED" if close else "OPEN", self._actuation)
        if self._actuation == "single_solenoid":
            self._io.set_digital_output(self._close_pin, bool(close), port=self._port)
        elif self._actuation == "single_toggle":
            # Every pulse flips the jaws whatever they are doing, so a pulse sent on a wrong
            # idea of where they stand moves them the wrong way. It is never held high: one
            # pulse is one flip.
            if self._toggle_stands_closed() != bool(close):
                self._pulse_toggle(close=bool(close))
                if not close and not self._await_open():
                    # The open switch measured that the jaws never reached the open stop,
                    # after a lost pulse or a stroke longer than close_timeout_s. Believing
                    # them open would report a held part as released.
                    self._closed = True
                    return
            else:
                self.logger.info("jaws already stand %s: no pulse", "CLOSED" if close else "OPEN")
        else:
            # A bistable valve latches, so the coil is energised only long enough to
            # throw it; holding it high is what cooks the coil. The opposite coil is
            # dropped first so both are never energised together, which on a real valve
            # stalls the spool.
            pin = self._close_pin if close else self._open_pin
            other = self._open_pin if close else self._close_pin
            assert pin is not None and other is not None  # narrowed by the constructor's refusal
            self._io.set_digital_output(other, False, port=self._port)
            self._io.set_digital_output(pin, True, port=self._port)
            self._sleep(self._pulse_s)
            self._io.set_digital_output(pin, False, port=self._port)
        self._closed = bool(close)

    def _pulse_toggle(self, *, close: bool) -> None:
        """One flip: the pin low for ``pulse_s``, the rising edge, high for ``pulse_s``, then low.

        The low comes first because a flip is an edge. A pin left high by a pulse whose
        low write failed, or two commands back to back, would otherwise give the device no
        edge while the driver believed in a flip. The belief moves at the edge, before the
        high is waited out, so an interrupt after it, such as Ctrl-C or a dropped link on
        the low write, leaves the driver agreeing with the jaws. Once the high write has
        been tried, the ``finally`` attempts the low, as the bench's own pulse does
        (``drivers/ur/io_bench.pulse_output``). A high write that raises may still have
        reached the controller with its reply lost, so whether the jaws flipped is
        unknowable: ``_edge_unknown`` stays set, and a cell with no open switch refuses
        its next pulse until a connect.
        """
        self._io.set_digital_output(self._close_pin, False, port=self._port)
        self._sleep(self._pulse_s)
        # The record says "on the way" before the edge and where the jaws stand after it, so a
        # program that dies between the two leaves the next one refusing rather than guessing.
        self._record(pulse_toward=close)
        self._edge_unknown = True
        try:
            self._io.set_digital_output(self._close_pin, True, port=self._port)
            self._closed = close
            self._edge_unknown = False
            self._record(stands_closed=close)
            self._awaiting_open = not close and self._open_confirm_pin is not None
            self._sleep(self._pulse_s)
        finally:
            self._io.set_digital_output(self._close_pin, False, port=self._port)

    def _record(self, *, pulse_toward: "bool | None" = None, stands_closed: "bool | None" = None) -> None:
        """Write the toggle's record, if it keeps one. A failed write is logged and does not stop the pulse.

        The jaws move whether or not the file could be written, and this program's own count stays right; what a
        failed write costs is the next program's start, which is what the error says.
        """
        if self._state_file is None:
            return
        by = f"pid {os.getpid()}"
        try:
            if pulse_toward is not None:
                jaw_toggle_state.record_pulse_started(self._state_file, toward_closed=pulse_toward, by=by)
            if stands_closed is not None:
                jaw_toggle_state.record_stands(self._state_file, closed=stands_closed, by=by)
        except OSError as exc:
            self.logger.error(
                "could not write the toggle record %s (%s): the next program may start from a stale record. "
                "Declare the jaws before running it if it does not open them as expected.", self._state_file, exc,
            )

    def _await_open(self) -> bool:
        """After a toggle's open pulse, wait for the open switch; whether the jaws got there.

        A close commanded while the jaws are still travelling open would read as not open
        and be swallowed, and the jaws would then finish opening under a driver that
        believes them closed. With no open switch there is nothing to wait on, and nothing
        measured otherwise, so this answers True; the next pulse's leading low is the gap.
        A timeout is logged and answered False: the switch measured that the jaws never
        got there. A lost pulse and a stroke longer than ``close_timeout_s`` cannot be told
        apart, so the timeout must be longer than the stroke.
        """
        if self._open_confirm_pin is None:
            return True
        started = time.monotonic()
        deadline = started + self._close_timeout_s
        while not bool(self._io.get_digital_input(self._open_confirm_pin, port=self._port)):
            if time.monotonic() >= deadline:
                self._awaiting_open = False
                self.logger.warning(
                    "jaws did not reach the open stop within %.2f s after the toggle's open pulse "
                    "(a lost pulse, or a stroke longer than close_timeout_s); taking them to "
                    "stand CLOSED",
                    self._close_timeout_s,
                )
                return False
            self._sleep(0.02)
        self._awaiting_open = False
        self.logger.info("jaws reached the open stop after %.3f s", time.monotonic() - started)
        return True

    def _settle_toggle_close(self) -> None:
        """After a toggle's close and its wait, follow an open switch that still reads open.

        It measured that the jaws never left, after a lost pulse or a stroke longer than
        the wait. Believing them closed would let a part-present sensor, which also sees a
        part between open jaws, report a grasp on jaws that never moved, so the belief
        follows the switch.
        """
        if self._open_confirm_pin is None:
            return
        if bool(self._io.get_digital_input(self._open_confirm_pin, port=self._port)):
            self.logger.warning(
                "jaws still read the open stop after the toggle's close pulse (a lost pulse, or "
                "a stroke longer than close_timeout_s); taking them to stand OPEN, no grasp "
                "is reported",
            )
            self._closed = False

    def _connect_toggle(self) -> None:
        """A toggle's connect, which sets the belief on every connect.

        The belief is never carried over from before a disconnect, because a person may
        have moved the jaws in between. With no open switch the jaws are taken to stand
        open and nothing is pulsed: every later pulse flips them from there, which is why
        the desk row says to stand them open with a pulse before connecting. With one, the
        switch decides. Open means no pulse. Closed, with the closed switch proving them
        empty, means one pulse opens them. Both switches active is warned about and never
        pulsed on. Anything else, off the open stop with no proof of empty, is refused
        (2026-09-23): a pulse there could drop a part wherever the arm is standing, and a
        switch that is not wired reads exactly that, after which every close would be
        swallowed and every open would close the jaws, each reported as done.
        """
        left_closed, unknowable = self._closed, self._edge_unknown
        self._closed, self._edge_unknown = False, False
        if bool(self._io.get_digital_output(self._close_pin, port=self._port)):
            # A pulse whose low write failed left the pin high. The device flips on the
            # rising edge, since a pulse is power on, so the low moves nothing, and without
            # it the next pulse would have no edge.
            self._io.set_digital_output(self._close_pin, False, port=self._port)
            self.logger.warning(
                "connect(): close pin=%s was still HIGH from an interrupted pulse; set it LOW",
                self._close_pin,
            )
        if self._open_confirm_pin is None:
            record = jaw_toggle_state.load(self._state_file) if self._state_file is not None else None
            if record is not None:
                self._start_from_record(record)
                return
            if unknowable or left_closed:
                # The same instance, connected again: disconnect never releases, and only a
                # person can have opened the jaws since. This is said at warning level,
                # because if nobody did, every command from here is inverted and nothing on
                # this wiring can notice.
                why = (
                    "the last pulse failed on its high write, so where the jaws stand is unknowable"
                    if unknowable else "the driver last took the jaws to stand CLOSED"
                )
                self.logger.warning(
                    "connect(): single_toggle with no open switch, and %s. Taking them to stand "
                    "OPEN: if they stand closed, every open and close from here is inverted. If "
                    "they do, %s before running.", why, self._open_with_a_pulse(),
                )
            self.logger.info(
                "connect(): single_toggle on %s close pin=%s with no open switch; taking the jaws "
                "to stand OPEN and not pulsing. Stand them open with a pulse before connecting, "
                "never by hand: every later pulse flips them from here, and a device that keeps "
                "its own flip state is not moved by a hand.", self._port.value, self._close_pin,
            )
            return
        if self._awaiting_open:
            self._await_open()  # an open interrupted mid-stroke finishes before the read
        state = self._read_state()
        if state is JawState.OPEN:
            self.logger.info(
                "connected on %s: single_toggle on close pin=%s; jaws read open -> no pulse",
                self._port.value, self._close_pin,
            )
            return
        if state is JawState.CONTRADICTORY:
            self.logger.warning(
                "connect(): both end-stop switches read active, which is mechanically impossible; "
                "check the wiring / sensors. not pulsing on an unreadable state.",
            )
            return
        self._closed = True
        if state is JawState.CLOSED_EMPTY:
            self._actuate(close=False)
            self.logger.info(
                "connected on %s: single_toggle on close pin=%s; jaws read closed on nothing -> "
                "one pulse sent, %s", self._port.value, self._close_pin,
                "and they stand CLOSED still" if self._closed else "and they reached the open stop",
            )
            return
        self._connected = False
        raise RobotError(
            f"JawIOGripper: the single_toggle's open switch, input {self._open_confirm_pin} on "
            f"{self._port.value}, does not read open, and nothing proves the jaws empty "
            f"({state.value}). Either they hold a part, and a pulse here would drop it wherever "
            f"the arm stands, or they stand open and the switch is not wired or reads inverted, "
            f"and then every close would be swallowed and every open would close them. Not "
            f"connecting: {self._open_with_a_pulse()}, check that input "
            f"{self._open_confirm_pin} reads high with the jaws open, and connect again."
        )

    def _start_from_record(self, record: "jaw_toggle_state.ToggleRecord") -> None:
        """A toggle with no open switch connects from where its record says the jaws stand.

        The record is this driver's own count of its pulses, carried from the last program, so a
        program that ended with the jaws closed no longer hands the next one an inverted count. A
        record of a pulse that was started and not seen to finish says nothing about the jaws, and
        the first pulse is then refused until a person declares where they stand.
        """
        self._record_in_flight = record.in_flight
        self._closed = record.closed and not record.in_flight
        declare = self._declare_command()
        if record.in_flight:
            self.logger.warning(
                "connect(): single_toggle with no open switch, and its record %s says %s. Where the jaws stand "
                "is unknown, and the first pulse is refused until a person has looked and declared it: %s",
                self._state_file, record.describe(), declare,
            )
            return
        if record.closed:
            self.logger.warning(
                "connect(): single_toggle with no open switch; its record says %s. Taking them to stand CLOSED, "
                "so the next open pulses once. If anyone has moved them since (the pendant's I/O tab, a power cut "
                "mid stroke), declare where they stand before running: %s", record.describe(), declare,
            )
            return
        self.logger.info("connect(): single_toggle with no open switch; its record says %s", record.describe())

    def _declare_command(self) -> str:
        """The command a person runs after looking at the jaws, said wherever the driver needs a declaration."""
        return (f"python -m src.robot.drivers.ur --profile <cell> --port {self._port.value} "
                f"--jaws-stand open (or closed) --yes")

    def _open_with_a_pulse(self) -> str:
        """How a toggle's jaws are stood open, said wherever the driver asks for it: a bench pulse.

        Never by hand. A device that keeps its own flip state is not moved by a hand that
        pushes its jaws open: its next pulse only brings that state round, nothing visibly
        moves, and every command after it is inverted, which is the leading explanation of
        the owner's cell closing at the tray (2026-09-23). A pulse keeps the device and the
        jaws in step on every kind of toggle.
        """
        return (f"open them with a pulse, never by hand (python -m src.robot.drivers.ur --profile "
                f"<cell> --port {self._port.value} --pulse {self._close_pin} --for {self._pulse_s} "
                f"--yes, once per flip, until one visibly opens them, then declare it: python -m "
                f"src.robot.drivers.ur --profile <cell> --port {self._port.value} --jaws-stand open --yes)")

    def _toggle_stands_closed(self) -> bool:
        """Whether the jaws stand closed, for a toggle deciding whether to pulse.

        The open switch says how the jaws stand where it is wired, and then it wins over
        the belief, which is what makes a lost pulse harmless. A lone closed switch cannot
        say it off its stop and is refused at construction. A part-present sensor does not
        say it either: it reads a part between open jaws as well as between closed ones,
        and between open jaws is where every close is commanded. With no open switch the
        driver can only trust what it last commanded, and it refuses when even that is
        unknowable, after a pulse whose high write failed. Both switches active raises
        rather than pulsing on a state nobody can read.
        """
        if self._open_confirm_pin is None:
            if self._record_in_flight:
                raise RobotError(
                    f"JawIOGripper: the toggle's record {self._state_file} says a pulse was started and not seen "
                    "to finish, so where the jaws stand is unknown and a pulse could move them either way. Not "
                    f"pulsing: look at the jaws and declare where they stand ({self._declare_command()}), then run "
                    "again."
                )
            if self._edge_unknown:
                raise RobotError(
                    "JawIOGripper: the last toggle pulse failed on its high write, which may "
                    "still have reached the controller, so whether the jaws flipped is unknowable "
                    "and the next pulse could move them either way. Not pulsing: look at the jaws, "
                    f"{self._open_with_a_pulse()} if they stand closed, and connect again."
                )
            return self._closed
        if self._awaiting_open:
            self._await_open()  # an open interrupted mid-stroke: travelling jaws read as closed
        state = self._read_state()
        if state is JawState.CONTRADICTORY:
            raise RobotError(
                "JawIOGripper: both end-stop switches read active, which is mechanically "
                "impossible, so nothing says how the jaws stand and a pulse could flip them either "
                "way. Not pulsing: check the wiring and the sensors."
            )
        if state is JawState.UNKNOWN:
            # With a switch wired, only a lone open switch answers UNKNOWN, and only off its
            # stop. Not open is all a toggle needs, and it is the switch saying it.
            return True
        return state is not JawState.OPEN

    def _read_state(self) -> JawState:
        """Classify the jaws from whichever inputs are wired. Never raises, never guesses."""
        closed_confirm = (
            None if self._closed_pin is None
            else bool(self._io.get_digital_input(self._closed_pin, port=self._port))
        )
        open_confirm = (
            None if self._open_confirm_pin is None
            else bool(self._io.get_digital_input(self._open_confirm_pin, port=self._port))
        )
        if closed_confirm is not None and open_confirm is not None:
            if closed_confirm and open_confirm:
                return JawState.CONTRADICTORY
            if open_confirm:
                return JawState.OPEN
            if closed_confirm:
                return JawState.CLOSED_EMPTY
            return JawState.HOLDING
        if closed_confirm is not None:
            # One switch, on the closed end stop. That is enough for the question that
            # matters: reaching the stop means the jaws met, so an empty grasp is
            # distinguishable. Short of it, whether the jaws are open or holding
            # depends on what was commanded.
            if closed_confirm:
                return JawState.CLOSED_EMPTY
            return JawState.HOLDING if self._closed else JawState.OPEN
        if open_confirm is not None:
            # One switch, on the open end stop. It can prove open and nothing else: off
            # the stop, holding a part and closed on nothing are the same reading. This
            # says UNKNOWN rather than picking one, because this is the wiring that
            # cannot detect an empty grasp, and a driver that pretended otherwise would
            # report every empty close as a successful grasp.
            return JawState.OPEN if open_confirm else JawState.UNKNOWN
        if self._part_pin is not None:
            if bool(self._io.get_digital_input(self._part_pin, port=self._port)):
                return JawState.HOLDING
            return JawState.CLOSED_EMPTY if self._closed else JawState.OPEN
        return JawState.UNKNOWN

    def _await_close(self) -> JawState:
        """Wait for the jaws to settle after a close and return the verdict. A timeout is not an error.

        A timeout means the jaws never reached a state the wiring can name, which for a
        grasp is the same class of event as a cup that never sealed: a missed grasp,
        decided by the verification stage. Raising would turn one that did not catch
        into a crash.

        Only the closed stop, reached, is a verdict at once, and only a closed switch can
        say it. Every other reading also occurs mid-stroke: a reed pair reads off both
        stops while the jaws travel, a lone closed switch off its stop repeats the command,
        and a part-present sensor sees a part between open jaws. Each was taken on the
        first read after the close until 2026-09-23, so the grasp returned and the arm
        lifted before the jaws had moved. Such a reading is taken after ``close_settle_s``,
        the jaws' travel time, and read again then.
        """
        if not self.has_feedback:
            # Nothing to poll. The cylinder still needs its travel time, and that wait
            # is the only thing separating a commanded valve from a gripped part.
            self._sleep(self._close_settle_s)
            return JawState.UNKNOWN
        started = time.monotonic()
        deadline = started + self._close_timeout_s
        travelled = False
        while True:
            state = self._read_state()
            proven = state is JawState.CLOSED_EMPTY and self._closed_pin is not None
            if state in (JawState.HOLDING, JawState.CLOSED_EMPTY) and not (proven or travelled):
                # A reading mid-stroke looks the same, so the jaws get their travel time first.
                self._sleep(self._close_settle_s)
                travelled = True
                continue
            if state in (JawState.HOLDING, JawState.CLOSED_EMPTY):
                # HOLDING against CLOSED_EMPTY is the grasp verdict, and the travel time
                # is what close_settle_s is budgeted from. Both are worth having per
                # attempt.
                self.logger.info(
                    "jaws settled: %s after %.3f s", state.value, time.monotonic() - started,
                )
                return state
            if time.monotonic() >= deadline:
                self.logger.info(
                    "jaws did not reach a settled state within %.2f s (last read: %s); treating as "
                    "no grasp (a missed grasp, not a fault)", self._close_timeout_s, state.value,
                )
                return state
            self._sleep(0.02)
