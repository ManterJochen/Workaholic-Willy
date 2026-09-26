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

A ``single_toggle`` is the exception, and the owner's cell (2026-09-24): a Hand-E on the
Robotiq I/O Coupling, one tool output, 24 V, where every short pulse flips the jaws and
nothing is wired back. It reads no sensor at all (a feedback pin is refused), takes no
width (``set_width_mm`` is refused; the verbs say open or close through ``set_closed``),
and counts its own pulses from where a person said the jaws stood when it connected:

* :meth:`JawIOGripper.connect` asks, once, before anything moves, whether the jaws stand
  open, and Enter answers open there. Closed is answered with one pulse to open them, or an
  abort. With no terminal and no question handed in, the connect is refused. The hand counts
  as connected, and its count starts, only once the answer is in.
* Every pulse flips the count, whatever verb sent it, and a pulse is sent only where the
  count says the jaws stand the other way from the request. The count flips only once the
  pin has read back HIGH; a pin that never does leaves the count unknowable, and the next
  pick asks.
* A pick never pulses before the arm moves; it asks :meth:`JawIOGripper.jaws_open_for_a_pick`,
  which asks the person again where the count says closed. There an empty line is no answer:
  only the word ``open`` or ``closed`` is.

At the terminal, what was typed before a question is discarded before it is asked, so a
stray Enter (examples 14 and 15 use Enter as push-to-talk) cannot answer it unseen. A
question whose connection changed while it waited, a disconnect or another connect from
another thread, is refused without a pulse. Nothing is kept between programs: the next
program asks again.

Status is bucket 3: this has never touched a real gripper. The I/O calls it makes are
the ones the UR driver already exposes. What is unverified is the wiring, meaning the
pin numbers, which port block, whether the reed switches are active-high, whether the
valve is single-acting, double-acting or a toggle that flips on every pulse, and how
long the cylinder takes to travel on the actual hardware. Every one of those is a
field on :class:`~src.config.schema.robot.robot_schema.JawIOGripperConfig`, so
bring-up is a matter of measuring numbers rather than editing this file.
"""

from __future__ import annotations

import importlib
import itertools
import sys
import threading
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

from src.robot.core import RobotConnectionError, RobotError
from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO
from src.robot.core.gripper import HoldEvidence

from ..constants import JAW_IO_GRIPPER_LOG_FILE, create_robot_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot.robot_schema import GripperConfig

__all__ = ["JawIOGripper", "JawQuestion", "JawState"]

#: The seam a person answers through: it is shown one question and returns what was typed, exactly as ``input``
#: does. The default where stdin is a terminal is ``input``, with the console's typeahead discarded before each
#: question (:func:`_drain_typeahead`). A test or a console hands in its own, which is never drained.
JawQuestion = Callable[[str], str]

#: How often an answer that is none of the choices is asked again before the question counts as unanswered.
_ASKS = 3

#: How often an output is read back after a write: one CB3 controller cycle, 125 Hz. The output goes out over RTDE
#: and its state comes back on the receive stream, so a read straight after a write reports the level before it
#: (measured on URSim 5.26.0, ``drivers/ur/io_bench.set_output``).
_READ_BACK_POLL_S = 0.008
#: The shortest a read-back waits, whatever the pulse: several controller cycles. A driver built directly, as the
#: tests build it, may pulse for 0 s; the config refuses a pulse under 0.05 s.
_READ_BACK_FLOOR_S = 0.05
#: The most keys one drain discards: a key held down repeats, and a drain must end.
_TYPEAHEAD_LIMIT = 4096


def _drain_typeahead(stdin: Any = None) -> int:
    """Discard what was typed at the console before a question is asked; how many keys were discarded, where countable.

    The question is read with the builtin ``input``, and ``input`` reads whatever waits in the
    console's input queue, typed before anybody asked. An Enter pressed while the models loaded
    (examples 14 and 15 use Enter as push-to-talk) answered the connect question unseen, and
    with the jaws really closed the count started OPEN and every command after it ran inverted
    (review of 2026-09-24, reproduced on a real Windows console). So the queue is emptied first,
    and only a key pressed after the question is shown can answer it.

    ``stdin`` is ``sys.stdin`` unless handed in. Only a stdin with a descriptor that is a terminal
    is drained: a pipe or a replaced stdin holds answers somebody meant, and no console. On
    Windows each waiting key is read off with ``msvcrt`` and counted; on POSIX the terminal's
    input queue is flushed (``termios.tcflush``, ``TCIFLUSH``), which counts nothing and returns
    0. A console that cannot be drained is asked as it stands: never raises.
    """
    stream = sys.stdin if stdin is None else stdin
    try:
        fd = int(stream.fileno())
        if not stream.isatty():
            return 0
    except (AttributeError, OSError, TypeError, ValueError):
        # No descriptor (a StringIO, a closed or replaced stdin): nothing typed at a console waits in it.
        return 0
    try:
        keyboard: Any = importlib.import_module("msvcrt")
    except ImportError:
        keyboard = None
    try:
        if keyboard is not None:
            drained = 0
            while drained < _TYPEAHEAD_LIMIT and keyboard.kbhit():
                keyboard.getwch()
                drained += 1
            return drained
        terminal: Any = importlib.import_module("termios")
        terminal.tcflush(fd, terminal.TCIFLUSH)
    except Exception:
        # A console handle gone, a terminal that refuses the flush: the question is asked as it stands, which is no
        # worse than before the drain existed, and a drain must never be why a question is not asked.
        return 0
    return 0


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

    ``confirm_open_at_start`` is whether :meth:`connect` asks a person where the jaws
    stand: ``None``, the default, asks for a ``single_toggle`` and not otherwise, and
    ``False`` is refused for a toggle. ``ask`` is how the question reaches the person
    (:data:`JawQuestion`); ``None`` asks at the terminal where stdin is one, through
    ``input``, with the console's typeahead discarded first.

    One lock holds a question and the pulse it leads to together: :meth:`connect`,
    :meth:`set_closed` and :meth:`jaws_open_for_a_pick` take it, so a command from another
    thread waits while a person is being asked, and never pulses in between. :meth:`disconnect`
    does not take it, because a teardown must never wait on a person; it ends the connection
    instead, and a question that finds its connection changed when its answer comes back is
    refused without a pulse.
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
        confirm_open_at_start: bool | None = None,
        min_width_mm: float = 0.0,
        max_width_mm: float = 85.0,
        ask: "JawQuestion | None" = None,
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
        if actuation == "single_toggle":
            # Every refusal below is the schema's, repeated because the constructor is reachable without it.
            if open_output_pin is not None:
                raise ValueError(
                    "JawIOGripper: actuation 'single_toggle' opens by a second pulse on "
                    "close_output_pin, so open_output_pin is never driven. Drop it, or use "
                    "'double_solenoid'."
                )
            if open_on_connect_without_feedback:
                raise ValueError(
                    "JawIOGripper: actuation 'single_toggle' cannot assert 'open' on connect: every "
                    "pulse flips the jaws, so one on jaws already open would close them. Drop "
                    "open_on_connect_without_feedback."
                )
            wired = [pin for pin in (part_present_input_pin, closed_confirm_input_pin, open_confirm_input_pin)
                     if pin is not None]
            if wired:
                # The toggle reads nothing back: the program counts its pulses from where a person said the jaws
                # stood, so an input here is a wire nobody reads, and a cell that believes it is sensed is not.
                raise ValueError(
                    f"JawIOGripper: actuation 'single_toggle' reads no sensor, so input {wired[0]} would be a "
                    "wire nobody reads: the program counts its own pulses from where a person said the jaws "
                    "stood at connect. Drop the input, or drive a valve that reads one as 'single_solenoid' or "
                    "'double_solenoid'."
                )
            if confirm_open_at_start is False:
                raise ValueError(
                    "JawIOGripper: actuation 'single_toggle' always asks at connect where its jaws stand: "
                    "nothing else can tell the program, and a count started from a wrong guess inverts every "
                    "command after it. Drop confirm_open_at_start: false."
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
        self._asks_at_start = actuation == "single_toggle" or bool(confirm_open_at_start)
        if config is not None:
            min_width_mm, max_width_mm = float(config.min_width_mm), float(config.max_width_mm)
        self._min_width_mm = float(min_width_mm)
        self._max_width_mm = float(max_width_mm)
        self._ask = ask
        self._sleep = sleep
        # True only once a connect has finished, its question answered: a hand still being asked about is not
        # connected, so nothing can command it (review of 2026-09-24, a race found in the console).
        self._connected = False
        # A question and the pulse it leads to, held together (see the class docstring). Reentrant, because the
        # person's answer may come from code that connects or commands on the same thread.
        self._lock = threading.RLock()
        # Which connection a question was asked on: a new one at every connect and every disconnect, so a question
        # whose answer comes back to a different one is refused rather than acted on.
        self._generations = itertools.count(1)
        self._generation = 0
        # What the driver last commanded, not what the jaws are doing. It is kept apart
        # from the sensed state on purpose: conflating the two is how a gripper reports
        # a grasp it never made. A single_toggle has no level to command, so for it this
        # is the program's count of its own pulses: set from a person's answer at every
        # connect, and flipped at each pulse's edge.
        self._closed = False
        # A toggle pulse whose high write failed, so whether the jaws flipped is
        # unknowable. The driver refuses to pulse again until a person has said where the
        # jaws stand, at the next pick's question or the next connect's.
        self._edge_unknown = False
        # Every command sent to the jaws since this driver was built, connects included; read
        # through `commands_sent`.
        self._commands_sent = 0
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
        program's count of its own pulses (see ``_closed``).
        """
        return self._closed

    @property
    def commands_sent(self) -> int:
        """How many commands this driver has sent the jaws since it was built, the ones a connect sent included.

        One per toggle pulse and one per solenoid actuation, counted once its commanding write
        (the rising edge, the level, the coil) has been tried: a write that raised may still have
        reached the controller, so the count never says less went out than may have. Never reset,
        not even by a connect: a caller that wants what one call sent reads it before and after.
        The bench's ``--jaws`` reads it so its summary says what went out across the connect and
        the command, and never "nothing was pulsed" where a connect's question sent a pulse.
        """
        return self._commands_sent

    @property
    def toggles_without_sensor(self) -> bool:
        """``True`` for a ``single_toggle``: every command is a pulse that flips the jaws, and nothing reads them."""
        return self._actuation == "single_toggle"

    @property
    def edge_unknown(self) -> bool:
        """``True`` once a pulse's high write failed or never read back: nobody can say whether the jaws flipped.

        Until a person says where the jaws stand again (the connect's question, or a pick-start re-ask), every
        pulse is refused. A caller that must never block on that question, such as a console run's thread,
        reads this and stops instead of asking.
        """
        return self._edge_unknown

    @property
    def asks_at_connect(self) -> bool:
        """Whether :meth:`connect` asks a person where the jaws stand: a ``single_toggle`` always, a solenoid opted in."""
        return self._asks_at_start

    # --- lifecycle --------------------------------------------------------
    def connect(self) -> None:
        """Adopt the arm's I/O and reach a safe state without blindly dropping a workpiece.

        This is the call that asks a person where the jaws stand, where one is asked:
        ``connect_cell`` (``src/robot/execution/lifecycle.py``) makes it when a program
        brings its cell up (``Robot.connected()``, ``Cell.connected()``; the bench's
        ``--jaws`` makes it itself), once, after the arm connects and before anything
        moves, and a refusal here rolls the arm back. A second call while connected does nothing; a call that
        raised left nothing connected, and the next one asks again. The hand counts as connected only
        once the question is answered and anything it chose has been sent: while it waits,
        :attr:`is_connected` is ``False`` and every command is refused, so no other thread can pulse
        jaws the count has not started on. A disconnect while it waits refuses the connect.

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
        part stays exactly as the last run left it. A ``single_toggle`` connects by its own
        rule, :meth:`_connect_toggle`, and a solenoid that opted into
        ``confirm_open_at_start`` asks first too: an answer of closed opens it only where
        the person chooses that, and an abort refuses the connect.
        """
        with self._lock:
            if self._connected:
                return
            self._generation = generation = next(self._generations)
            # A connect that raises sets nothing: it never marked the hand connected, so the next one starts again,
            # question included.
            if self._actuation == "single_toggle":
                self._connect_toggle()
            else:
                self._connect_solenoid()
            if self._generation != generation:
                # Past the question, which checks this itself: a disconnect came while the pulse the person chose, or
                # its stroke, ran. Marking the hand connected now would undo that disconnect.
                raise RobotError(
                    f"JawIOGripper: not connecting: the hand on {self._where_pin()} was disconnected while its connect "
                    "ran. Connect again, and it asks again where the jaws stand."
                )
            self._connected = True

    def _connect_solenoid(self) -> None:
        """A solenoid's connect, by the rule :meth:`connect` states; it asks first where it opted in."""
        if self._asks_at_start:
            self._refuse_unless_open("this cell asks where they stand before anything moves (confirm_open_at_start)")
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
        explicit ``set_closed(False)`` call, so a caller that wants it asks for it.

        It takes no lock, so it never waits on a person: it ends the connection, and a question
        still waiting on this one finds that when its answer comes back and pulses nothing.
        """
        self._connected = False
        self._generation = next(self._generations)
        # This says what was left behind, because the driver deliberately does not
        # release: a cell found holding a part next morning is explained by this line.
        left = ("UNKNOWN, the last pulse failed on its high write or never read HIGH" if self._edge_unknown
                else "CLOSED" if self._closed else "open")
        if self._actuation == "single_toggle":
            self.logger.info("disconnected without releasing (jaws left %s); the next connect asks where they "
                             "stand", left)
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

        Refused for a ``single_toggle``, which takes no width: a width read against a
        threshold is how a grasp became an open on the owner's cell (2026-09-23), and a
        toggle has nothing to measure it against. Its verbs say open or close through
        :meth:`set_closed`.
        """
        self._require_connected("set_width_mm")
        if self._actuation == "single_toggle":
            raise RobotError(
                f"JawIOGripper: a single_toggle takes no width ({float(width_mm)} mm was sent): every pulse flips "
                "the jaws and nothing measures them. Say open or close with set_closed(), as the hand verbs do."
            )
        self.set_closed(float(width_mm) <= self._closed_below_mm)

    def set_closed(self, closed: bool) -> None:
        """Close or open the jaws by intent, whatever ``closed_below_mm`` says (``OpensAndCloses``).

        The width a caller sends is read against ``closed_below_mm``, and a grasp above it
        is an open (owner's cell, 2026-09-23). A verb that knows what it means, a pick's
        pre-open, its grasp, a release, says it here instead, so no width can turn it round.
        A close waits for the jaws exactly as a closing width does.

        A ``single_toggle`` sends one pulse where its count says the jaws stand the other
        way, and then waits ``close_settle_s``, the stroke, whichever way it went. Where they
        already stand there it sends nothing and says so in the log.

        It holds the lock, so it waits while a person is being asked where the jaws stand.
        """
        with self._lock:
            self._require_connected("set_closed")
            close = bool(closed)
            if self._actuation == "single_toggle":
                if self._toggle(close):
                    self._sleep(self._close_settle_s)
                return
            was_closed = self._closed
            self._actuate(close=close)
            if close:
                self._await_close()
            elif was_closed and self._open_confirm_pin is None:
                # The jaws travel the same stroke open as closed, and with no open switch the travel time is the only
                # wait there is. Without it a place returned the moment the valve was told, and the arm backed out of
                # a Hand-E that takes up to about 2 s to open, dragging the part it had just set down (owner's cell,
                # 2026-09-23).
                self._sleep(self._close_settle_s)

    def jaws_open_for_a_pick(self) -> str:
        """Before a pick moves the arm: ``""`` where the jaws stand open, else why the pick must not start.

        ``TogglesWithoutSensor``: a pick on a toggle never pulses before the arm moves. Where
        the count says open this answers at once. Where it says closed (a pick with no place
        before it) or cannot say (a pulse whose high write failed, or whose pin never read
        HIGH), it asks the person again, as :meth:`connect` does, and a person who answers
        closed may choose one pulse to open them now. Here an empty line is no answer: the
        program already believes something else, so only the word ``open`` or ``closed`` is,
        and an Enter meant for something else is asked again. With nobody to ask, the pick is
        refused. A question whose connection changed while it waited is refused without a
        pulse. Every other actuation answers ``""``: its pick opens the jaws by command.
        """
        with self._lock:
            self._require_connected("jaws_open_for_a_pick")
            if self._actuation != "single_toggle" or not (self._closed or self._edge_unknown):
                return ""
            why = ("a pick starts with them open, and the last pulse failed on its high write or its pin never read "
                   "HIGH, so nobody can say which way it moved them" if self._edge_unknown else
                   "a pick starts with them open, and the program believes they stand CLOSED: its last command "
                   "closed them")
            refused = self._where_the_jaws_stand(why, at_connect=False)
            return f"not starting the pick: {refused}" if refused else ""

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
        if not self._closed:
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
        jaws hold nothing and the reeds read between the stops during travel. Closed, a part
        pin that reads high is HELD and low is EMPTY. With no part pin, ``JawState.HOLDING``
        is HELD, ``JawState.CLOSED_EMPTY`` is EMPTY, and any other reed state, or no reeds at
        all, is UNMEASURED: a ``single_toggle``, which reads nothing, is always UNMEASURED.
        """
        self._require_connected("hold_evidence")
        if not self._closed:
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

    def read_jaw_state(self) -> JawState:
        """The sensed jaw state, for operators and bring-up. Not part of any Protocol."""
        self._require_connected("read_jaw_state")
        return self._read_state()

    # --- internals --------------------------------------------------------
    def _require_connected(self, what: str) -> None:
        if not self._connected:
            raise RobotConnectionError(f"JawIOGripper.{what} requires connect() first.")

    def _actuate(self, *, close: bool) -> None:
        """Drive a solenoid valve: a single one holds a level, a double one pulses the right coil.

        Each commanding write is read back, as the toggle's is (:meth:`_reads_back`): a level
        or a coil that never reads back is refused with a :class:`RobotError` rather than
        believed. ``_closed`` is what was last commanded, set once the commanding write has
        returned, because the next command sets its own level or coil whatever this one did,
        so a solenoid has no count to invert; the raise is what says the pin never showed it.
        A double solenoid's coil is dropped whatever happens once it has been driven.
        """
        # Info, not debug: the robot logger keeps info, and whether a command drove the jaws at
        # all is the first thing a cell whose jaws did the wrong thing asks of its log (owner's
        # cell, 2026-09-23).
        self.logger.info("actuating jaws %s via %s", "CLOSED" if close else "OPEN", self._actuation)
        if self._actuation == "single_solenoid":
            self._commands_sent += 1
            self._io.set_digital_output(self._close_pin, bool(close), port=self._port)
            self._closed = bool(close)
            if not self._reads_back(self._close_pin, bool(close)):
                raise RobotError(self._never_read(self._close_pin, bool(close), "the jaws may not have moved"))
            return
        # A bistable valve latches, so the coil is energised only long enough to
        # throw it; holding it high is what cooks the coil. The opposite coil is
        # dropped first so both are never energised together, which on a real valve
        # stalls the spool.
        pin = self._close_pin if close else self._open_pin
        other = self._open_pin if close else self._close_pin
        assert pin is not None and other is not None  # narrowed by the constructor's refusal
        self._io.set_digital_output(other, False, port=self._port)
        self._commands_sent += 1
        try:
            self._io.set_digital_output(pin, True, port=self._port)
            self._closed = bool(close)
            if not self._reads_back(pin, True):
                raise RobotError(self._never_read(pin, True, "the valve may not have thrown"))
            self._sleep(self._pulse_s)
        finally:
            # An interrupt or a coil that never read back must not leave it energised: that is what cooks it.
            self._io.set_digital_output(pin, False, port=self._port)

    def _reads_back(self, pin: int, level: bool) -> bool:
        """Whether output ``pin`` reads ``level`` within :meth:`_read_back_s`, read once per controller cycle.

        A write that returns says the controller accepted the command, not that the pin moved: a
        wrong bank, a pin the controller reserves and a write that never reached the pin all look
        like success from here, which is why ``drivers/ur/io_bench.set_output`` reads back too.
        The state comes back on the RTDE receive stream a cycle or two after the write, so a first
        read that disagrees is read again every :data:`_READ_BACK_POLL_S` until the bound. The
        bound counts the waits asked of ``sleep``, so a test's sleep drives it exactly and a real
        one waits at least that long. A pin that agrees at once costs no wait.
        """
        bound = self._read_back_s()
        waited = 0.0
        while bool(self._io.get_digital_output(pin, port=self._port)) != level:
            if waited >= bound:
                return False
            step = min(_READ_BACK_POLL_S, bound - waited)
            self._sleep(step)
            waited += step
        return True

    def _read_back_s(self) -> float:
        """How long a write is read back before it counts as never shown: a pulse and two cycles, 50 ms at least."""
        return max(_READ_BACK_FLOOR_S, self._pulse_s + 2 * _READ_BACK_POLL_S)

    def _never_read(self, pin: int, level: bool, consequence: str) -> str:
        """The refusal for an output that never read back ``level``, naming the bank and the pin as the pendant does."""
        said = "HIGH" if level else "LOW"
        return (
            f"JawIOGripper: {self._port.value} output {pin} was set {said} and the output never read {said} within "
            f"{self._read_back_s():.3f} s, so {consequence}. A wrong bank (io_port), a pin the controller reserves, "
            "or a write that never reached the pin reads so; check the pin on the pendant's I/O tab."
        )

    def _toggle(self, close: bool) -> bool:
        """A toggle's command: one pulse where the count says the jaws stand the other way; whether it pulsed.

        Every pulse flips the jaws whatever they are doing, so a pulse sent on a wrong count
        moves them the wrong way. After a pulse whose high write failed, or whose pin never
        read HIGH, nobody can say which way the next one would move them, so this refuses until
        a person has said.
        """
        self.logger.info("actuating jaws %s via single_toggle", "CLOSED" if close else "OPEN")
        if self._edge_unknown:
            raise RobotError(
                f"JawIOGripper: the last pulse on {self._where_pin()} failed on its high write, which may still have "
                "reached the controller, or its pin never read HIGH, so whether the jaws flipped is unknowable and "
                "the next pulse could move them either way. Not pulsing: the next pick asks where they stand, or "
                "connect again."
            )
        if self._closed == close:
            self.logger.info("jaws already stand %s: no pulse", "CLOSED" if close else "OPEN")
            return False
        self._pulse_toggle()
        return True

    def _pulse_toggle(self) -> None:
        """One flip: the pin low for ``pulse_s``, the rising edge, high for ``pulse_s``, then low.

        The low comes first because a flip is an edge. A pin left high by a pulse whose
        low write failed, or two commands back to back, would otherwise give the device no
        edge while the driver believed in a flip. So the low is read back before the high
        is sent: a pin that still reads HIGH would give no edge, and nothing is sent.

        The count flips at the edge, and the edge is the pin read back HIGH, not the high
        write returning: a write that returns says the controller accepted it, not that the
        pin moved (:meth:`_reads_back`, review of 2026-09-24). A pin that never reads HIGH
        within :meth:`_read_back_s` flips nothing, leaves ``_edge_unknown`` set and raises,
        so the next pick asks where the jaws stand. Once read HIGH the count flips before
        the high is waited out, so an interrupt after it, such as Ctrl-C or a dropped link
        on the low write, leaves the count agreeing with the jaws. Once the high write has
        been tried, the ``finally`` attempts the low, as the bench's own pulse does
        (``drivers/ur/io_bench.pulse_output``). A high write that raises may still have
        reached the controller with its reply lost, so whether the jaws flipped is
        unknowable: ``_edge_unknown`` stays set, and the driver refuses its next pulse
        until a person has said where the jaws stand.
        """
        pin = self._close_pin
        self._io.set_digital_output(pin, False, port=self._port)
        self._sleep(self._pulse_s)
        if not self._reads_back(pin, False):
            # Nothing high was sent, so the count still agrees with the jaws: this pulse would only have had no edge.
            raise RobotError(self._never_read(pin, False, (
                "a pulse now would have no rising edge to flip the jaws on. Not pulsing, and nothing was sent: the "
                "count stands")))
        self._edge_unknown = True
        self._commands_sent += 1
        try:
            self._io.set_digital_output(pin, True, port=self._port)
            if not self._reads_back(pin, True):
                raise RobotError(self._never_read(pin, True, (
                    "nobody can say whether the jaws flipped. The pulse is not counted: the next pick asks where they "
                    "stand, or connect again")))
            self._closed = not self._closed
            self._edge_unknown = False
            self._sleep(self._pulse_s)
        finally:
            self._io.set_digital_output(pin, False, port=self._port)

    def _connect_toggle(self) -> None:
        """A toggle's connect: a pin left high is dropped, then a person says where the jaws stand.

        The count is never carried over from before, a disconnect or another program,
        because a person or the pendant may have moved the jaws in between; only a person
        looking at them can start it, and it starts only once the answer is in. A refusal
        leaves the gripper disconnected, and ``connect_cell`` rolls the arm back.
        """
        if bool(self._io.get_digital_output(self._close_pin, port=self._port)):
            # A pulse whose low write failed left the pin high. The device flips on the
            # rising edge, since a pulse is power on, so the low moves nothing, and without
            # it the next pulse would have no edge.
            self._io.set_digital_output(self._close_pin, False, port=self._port)
            self.logger.warning(
                "connect(): close pin=%s was still HIGH from an interrupted pulse; set it LOW",
                self._close_pin,
            )
        self._refuse_unless_open("every pulse flips them and nothing reads them back, so only a person can say where "
                                 "the program's count of its pulses starts")
        self.logger.info("connected on %s: single_toggle on close pin=%s; the jaws stand OPEN, and the program "
                         "counts its pulses from here", self._port.value, self._close_pin)

    def _refuse_unless_open(self, reason: str) -> None:
        """At connect: ask where the jaws stand, and refuse the connect unless they stand open by the end of it."""
        refused = self._where_the_jaws_stand(reason, at_connect=True)
        if refused:
            raise RobotError(f"JawIOGripper: not connecting: {refused}")

    def _where_the_jaws_stand(self, reason: str, *, at_connect: bool) -> str:
        """Ask a person whether the jaws stand open; ``""`` where they do by the end, else the one-line refusal.

        ``reason`` says why the question is asked, in the question and in a refusal. ``open``:
        they stand open, and the count starts there. ``closed``: the person chooses ``p``, one
        command now that opens them (a single pulse on a toggle, which releases whatever is
        between them), and the stroke is waited out, or ``a``, which aborts. Nothing moves
        without that choice. An answer that is none of these is asked again, and after
        :data:`_ASKS` of them, or at end of input, nothing is taken for granted: the answer is
        an abort.

        Enter answers open only ``at_connect``, the owner's agreed answer at program start. At a
        pick start the program already believes the jaws stand closed, or cannot say, so an
        empty line there, which an Enter meant for something else also gives, is no answer: only
        the word is, and the question says so.

        The answer is acted on only where the connection it was asked on still stands when it
        comes back: a disconnect or another connect meanwhile (another thread, which the lock
        does not hold off a disconnect) makes it an answer about a count nobody keeps any more,
        and it is refused with nothing sent.
        """
        where = self._where_pin()
        ask = self._question()
        if ask is None:
            return (f"nobody can be asked where the jaws on {where} stand ({reason}): stdin is not a terminal and "
                    "no question was handed to the gripper; run it from a terminal")
        asked_on = (self._generation, self._connected)
        if at_connect:
            question = (f"\nThe jaws of the {self._actuation} hand on {where}: {reason}.\n"
                        "Look at them. Do they stand OPEN? [Enter or 'open' = open, 'closed' = closed]: ")
            choices = {"": "open", "o": "open", "open": "open", "c": "closed", "closed": "closed"}
        else:
            question = (f"\nThe jaws of the {self._actuation} hand on {where}: {reason}.\n"
                        "Look at them. Do they stand OPEN? [type 'open' or 'closed'; Enter alone is no answer here]: ")
            choices = {"o": "open", "open": "open", "c": "closed", "closed": "closed"}
        answer = self._answer(ask, question, choices)
        changed = self._changed_while_asked(asked_on, reason)
        if changed:
            return changed
        if answer == "open":
            self._closed, self._edge_unknown = False, False
            self.logger.warning("a person said the jaws on %s stand OPEN (%s)", where, reason)
            return ""
        if answer is None:
            return f"no answer said where the jaws on {where} stand ({reason})"
        choice = self._answer(ask, (f"They stand CLOSED. [p] open them now with {self._how_it_opens()}, which "
                                    "releases anything between them; [a] abort: "),
                              {"p": "pulse", "pulse": "pulse", "a": "abort", "abort": "abort"})
        changed = self._changed_while_asked(asked_on, reason)
        if changed:
            return changed
        if choice != "pulse":
            # An abort is a choice; three wrong answers or the end of input are not, and the refusal says which it was.
            said = "chose not to open them" if choice == "abort" else "gave no clear answer to whether to open them"
            self.logger.warning("a person said the jaws on %s stand CLOSED and %s (%s)", where, said, reason)
            return f"a person said the jaws on {where} stand CLOSED and {said}"
        self.logger.warning("a person said the jaws on %s stand CLOSED and had them opened (%s)", where, reason)
        self._closed, self._edge_unknown = True, False
        if self._actuation == "single_toggle":
            self._toggle(False)
        else:
            self._actuate(close=False)
        self._sleep(self._close_settle_s)
        return ""

    def _changed_while_asked(self, asked_on: "tuple[int, bool]", reason: str) -> str:
        """``""`` where the connection a question was asked on still stands, else the refusal that pulses nothing."""
        if (self._generation, self._connected) == asked_on:
            return ""
        where = self._where_pin()
        self.logger.warning("the connection of the hand on %s changed while the question waited (%s); its answer is "
                            "not acted on and nothing was sent", where, reason)
        return (f"the connection of the hand on {where} changed while the question waited (a disconnect or another "
                "connect came meanwhile), so its answer is about a count nobody keeps any more; nothing was sent")

    def _answer(self, ask: "JawQuestion", question: str, choices: "dict[str, str]") -> "str | None":
        """What the person chose among ``choices``, asked up to :data:`_ASKS` times; ``None`` for no clear answer.

        An answer that is none of the choices, an empty line included where Enter is not one, is
        asked again with the question it answered, prefixed with why.
        """
        asking = question
        for _ in range(_ASKS):
            try:
                said = str(ask(asking)).strip().lower()
            except EOFError:
                return None
            if said in choices:
                return choices[said]
            why = "An empty line is no answer here." if not said else f"'{said}' is none of the choices."
            asking = f"{why} {question.lstrip()}"
        return None

    def _question(self) -> "JawQuestion | None":
        """The seam the person answers through: the one handed in, else the terminal where stdin is one."""
        if self._ask is not None:
            return self._ask
        try:
            interactive = sys.stdin is not None and sys.stdin.isatty()
        except (AttributeError, ValueError):  # a closed or replaced stdin
            interactive = False
        return self._ask_at_the_terminal if interactive else None

    def _ask_at_the_terminal(self, question: str) -> str:
        """The terminal's question: what was typed before it is discarded (:func:`_drain_typeahead`), then ``input``.

        Every question at the terminal is drained, the follow-up included, so only a key pressed
        after a question is shown answers it. A question handed in (``ask=``) owns its input and
        is never drained.
        """
        drained = _drain_typeahead()
        if drained:
            self.logger.warning("discarded %d key(s) typed at the console before the question about the jaws on %s: "
                                "only a key pressed after it answers it", drained, self._where_pin())
        return input(question)

    def _where_pin(self) -> str:
        """The bank and pin the jaws are driven on, as a person reads it on the pendant: ``tool output 0``."""
        return f"{self._port.value} output {self._close_pin}"

    def _how_it_opens(self) -> str:
        """The one command that opens the jaws, named for the question."""
        if self._actuation == "single_toggle":
            return f"one pulse on {self._where_pin()}"
        if self._actuation == "double_solenoid":
            return f"a pulse on {self._port.value} output {self._open_pin}"
        return f"{self._where_pin()} set low"

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
