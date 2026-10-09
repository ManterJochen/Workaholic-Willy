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
post-close verification in ``GraspExecutionPolicy``: the policy reads the hold right
after the close, as it reads a Robotiq's gOBJ and a vacuum switch.

A ``single_toggle`` is the exception, and the owner's cell: a Hand-E on the Robotiq I/O
Coupling, one tool output, 24 V, nothing wired back, where EVERY CHANGE of the output moves
the jaws once, switched on as much as switched off (the owner at the pendant, 2026-09-28). A
pulse, on and off, therefore moves them twice: the pulse this driver sent until then closed,
opened and closed the jaws at the part, and closed them again after every release. It reads
no sensor at all (a feedback pin is refused), takes no width (``set_width_mm`` is refused; the
verbs say open or close through ``set_closed``), and counts its own changes from where a
person said the jaws stood when it connected:

* :meth:`JawIOGripper.connect` writes nothing: it reads the output and asks, once, before
  anything moves, whether the jaws stand open, and Enter answers open there. Closed is
  answered with one change to open them, or an abort. With no terminal and no question
  handed in, the connect is refused. The hand counts as connected, and its count starts,
  only once the answer is in.
* A command is ONE change of the output, which is then left where it went, and it is sent
  only where the count says the jaws stand the other way from the request. The count flips
  only once the output has read back the new level; one that never does leaves the count
  unknowable, and the next pick asks.
* The output is read before every command. One that no longer stands where the program left
  it was switched by somebody else, at the pendant above all, and the count no longer says
  where the jaws stand: the command is refused, and the next pick asks.
* A pick never switches the output before the arm moves; it asks
  :meth:`JawIOGripper.jaws_open_for_a_pick`, which asks the person again where the count says
  closed or cannot say. There an empty line is no answer: only the word ``open`` or ``closed`` is.

At the terminal, what was typed before a question is discarded before it is asked, so a
stray Enter (examples 14 and 15 use Enter as push-to-talk) cannot answer it unseen. A
question whose connection changed while it waited, a disconnect or another connect from
another thread, is refused without a change. Nothing is kept between programs: the next
program asks again.

The operator console asks in the browser instead (build plan, item 10): it hands the hand a
structured seam (:meth:`JawIOGripper.answer_questions_with`), which gets one :class:`JawAsking` per
attempt, stage by stage, and acts only on a word it offered, and a gate, ``before_change``, asked
under the lock right before the one change a person chose, which reads the controller: a stopped arm
gets no "open now", and nothing is sent. Only the gate's empty sentence lets the change out, and the
connection is read again after it, so a disconnect while the gate reads sends nothing either. A
question nobody answers there is refused, never taken for "open", and a Restart asks every time
(:meth:`JawIOGripper.confirm_where_the_jaws_stand`): where the jaws stand, or, where the program's own
count says CLOSED and nobody switched the output since, only whether to open them now with one change,
so no answer of "open" can turn a count round that a part is still clamped against (review of
2026-10-02). CLI programs ask at their terminal as before.

The solenoids are bucket 3: they have never touched a real gripper. The I/O calls they make
are the ones the UR driver already exposes. What is unverified is the wiring, meaning the
pin numbers, which port block, whether the reed switches are active-high, whether the
valve is single-acting or double-acting, and how long the cylinder takes to travel on the
actual hardware. Every one of those is a field on
:class:`~src.config.schema.robot.robot_schema.JawIOGripperConfig`, so bring-up is a matter
of measuring numbers rather than editing this file.
"""

from __future__ import annotations

import contextlib
import importlib
import itertools
import sys
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Iterator

from src.robot.core import RobotConnectionError, RobotError
from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO, end_pulse
from src.robot.core.gripper import HoldEvidence

from ..constants import JAW_IO_GRIPPER_LOG_FILE, create_robot_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot.robot_schema import GripperConfig

__all__ = ["BeforeChange", "JawAsking", "JawIOGripper", "JawQuestion", "JawState", "StructuredJawQuestion"]

#: The seam a person answers through: it is shown one question and returns what was typed, exactly as ``input``
#: does. The default where stdin is a terminal is ``input``, with the console's typeahead discarded before each
#: question (:func:`_drain_typeahead`). A test or a console hands in its own, which is never drained.
JawQuestion = Callable[[str], str]

#: How often an answer that is none of the choices is asked again before the question counts as unanswered.
_ASKS = 3


@dataclass(frozen=True, slots=True)
class JawAsking:
    """One question a toggle hand asks through a structured seam (:meth:`JawIOGripper.answer_questions_with`), per
    attempt.

    ``stage`` is ``where`` (do the jaws stand open or closed?) or ``open_now``, asked after "closed", and first at a
    check where the program's own count says CLOSED: open them now with the one command that opens them, which releases
    whatever is between them, or abort. ``choices`` are the
    answers that stage takes, in the console's words: ``open`` and ``closed``, then ``open_now`` and ``abort``. Through
    the seam the hand takes exactly these, and for ``open_now`` also the build plan's own letters for them, ``p`` and
    ``a``; the terminal's ``o``, ``c`` and ``pulse`` are none of the choices there. There is no default: an empty
    answer is no answer, at connect too, and ``text`` says so. ``at_connect`` is true where a connect asks; otherwise a
    check (:meth:`JawIOGripper.confirm_where_the_jaws_stand`) or a pick start does. ``reason`` is why the hand asks,
    ``where`` the bank and pin as the pendant shows them (``tool output 0``), ``text`` the question in the hand's own
    words (the terminal's, less a default the seam does not offer), ``attempt`` which attempt this is of ``of``, and
    ``why_again`` why it is asked again (``""`` the first time).
    """

    stage: str
    at_connect: bool
    reason: str
    where: str
    choices: tuple[str, ...]
    text: str
    attempt: int
    why_again: str
    of: int = _ASKS


#: The seam a console answers through: handed one :class:`JawAsking`, it returns one of its ``choices``. ``EOFError``
#: (a timeout, a cancel, a disconnect) is no answer, and a question nobody answered is refused, never taken for "open".
StructuredJawQuestion = Callable[[JawAsking], str]
#: A gate asked under the hand's lock immediately before the one change a person chose to open the jaws: ``""`` lets it
#: go out, anything else refuses it with nothing sent and is said in the refusal, an answer that is no sentence at all
#: (``None``, ``False``, a blank) included. The console reads the controller here, so a stopped arm gets no "open now"
#: (the order of the pick service: the controller before the hand).
BeforeChange = Callable[[], str]

#: The answers a question's two stages take through a structured seam, in the console's words.
_WHERE_CHOICES = ("open", "closed")
_OPEN_NOW_CHOICES = ("open_now", "abort")
#: What a structured seam's answer may be, per stage: the choices it was offered, and for ``open_now`` the build plan's
#: own letters for them (``p``, ``a``). The terminal's ``o``, ``c`` and ``pulse`` were never offered there.
_SEAM_WORDS = {"where": frozenset(_WHERE_CHOICES), "open_now": frozenset((*_OPEN_NOW_CHOICES, "p", "a"))}

#: How often an output is read back after a write: one CB3 controller cycle, 125 Hz. The output goes out over RTDE
#: and its state comes back on the receive stream, so a read straight after a write reports the level before it
#: (measured on URSim 5.26.0, ``drivers/ur/io_bench.set_output``).
_READ_BACK_POLL_S = 0.008
#: The shortest a read-back waits, whatever the pulse: several controller cycles. A driver built directly, as the
#: tests build it, may pulse for 0 s; the config refuses a pulse under 0.05 s.
_READ_BACK_FLOOR_S = 0.05
#: How long a toggle's change is read back before it counts as never shown: about thirty CB3 controller cycles, where
#: a change reads back within two. It is the bound the default 0.2 s pulse gave the toggle before it stopped pulsing,
#: rounded up, and it does not hang on ``pulse_s``, which a toggle no longer uses.
_CHANGE_READ_BACK_S = 0.25
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


def _gate_said(answered: object) -> str:
    """What a gate's answer (:data:`BeforeChange`) says, in one line: ``""`` for the empty sentence alone, which lets
    the change go out; any other sentence's words; and why an answer that is no sentence (``None``, ``False``, a
    number, a blank) refuses too: a gate that forgot its return fails closed, never open (review of 2026-10-01)."""
    if isinstance(answered, str):
        if answered == "":
            return ""
        words = " ".join(answered.split())
        if words:
            return words
    return f"the check before the change answered no sentence ({answered!r}), and only an empty one lets it go out"


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
    ``input``, with the console's typeahead discarded first. A structured seam handed in later
    (:meth:`answer_questions_with`) comes before both, with the gate asked before every change.

    One lock holds a question and the command it leads to together: :meth:`connect`,
    :meth:`set_closed` and :meth:`jaws_open_for_a_pick` take it, so a command from another
    thread waits while a person is being asked, and never switches in between. :meth:`disconnect`
    does not take it, because a teardown must never wait on a person; it ends the connection
    instead, and a question that finds its connection changed when its answer comes back is
    refused with nothing sent.
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
                "or 'single_toggle' (one output, each change of it moves the jaws)."
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
                    "JawIOGripper: actuation 'single_toggle' opens by the next change of "
                    "close_output_pin, so open_output_pin is never driven. Drop it, or use "
                    "'double_solenoid'."
                )
            if open_on_connect_without_feedback:
                raise ValueError(
                    "JawIOGripper: actuation 'single_toggle' cannot assert 'open' on connect: every "
                    "change of its output moves the jaws, so one on jaws already open would close them. Drop "
                    "open_on_connect_without_feedback."
                )
            wired = [pin for pin in (part_present_input_pin, closed_confirm_input_pin, open_confirm_input_pin)
                     if pin is not None]
            if wired:
                # The toggle reads nothing back: the program counts its changes from where a person said the jaws
                # stood, so an input here is a wire nobody reads, and a cell that believes it is sensed is not.
                raise ValueError(
                    f"JawIOGripper: actuation 'single_toggle' reads no sensor, so input {wired[0]} would be a "
                    "wire nobody reads: the program counts its own changes from where a person said the jaws "
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
        # The console's structured seam and the gate it hands in with it (`answer_questions_with`): when set, every
        # question goes through the seam, before `ask` and the terminal, and the gate is asked before every change a
        # question leads to.
        self._structured: StructuredJawQuestion | None = None
        self._before_change: BeforeChange | None = None
        self._sleep = sleep
        # True only once a connect has finished, its question answered: a hand still being asked about is not
        # connected, so nothing can command it (review of 2026-09-24, a race found in the console).
        self._connected = False
        # A question and the change it leads to, held together (see the class docstring). Reentrant, because the
        # person's answer may come from code that connects or commands on the same thread.
        self._lock = threading.RLock()
        # Which connection a question was asked on: a new one at every connect and every disconnect, so a question
        # whose answer comes back to a different one is refused rather than acted on.
        self._generations = itertools.count(1)
        self._generation = 0
        # What the driver last commanded, not what the jaws are doing. It is kept apart
        # from the sensed state on purpose: conflating the two is how a gripper reports
        # a grasp it never made. A single_toggle's level says nothing on its own, so for it
        # this is the program's count of its own changes: set from a person's answer at
        # every connect, and flipped at each change the output read back.
        self._closed = False
        # A toggle whose count nobody can vouch for: a change whose write failed or never
        # read back, or an output somebody switched by hand. The driver refuses to switch
        # again until a person has said where the jaws stand, at the next pick's question or
        # the next connect's; `_unknown_why` says which it was.
        self._edge_unknown = False
        self._unknown_why = ""
        # Where a toggle's output stood when the count last agreed with it: read once a
        # person answered, and set by every change that read back. None before the first
        # answer. An output found elsewhere was switched by somebody else.
        self._level: bool | None = None
        # Why nobody is asked on the current thread, set by :meth:`asking_nobody`: a question the driver would ask on
        # that thread is a refusal instead. Per thread, so a caller that must never wait on a person (a run started
        # from the console) does not stop another thread's connect from asking.
        self._nobody_asked = threading.local()
        # Every command sent to the jaws since this driver was built, connects included; read
        # through `commands_sent`.
        self._commands_sent = 0
        # When a stroke `set_closed(wait=False)` left to its caller is over, `time.monotonic()` seconds, or None: the
        # next change waits it out first (`_stroke_left_out`).
        self._settles_at: float | None = None
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

        For a ``single_toggle``, whose every change moves the jaws whatever level it goes to,
        this is the program's count of its own changes (see ``_closed``).
        """
        return self._closed

    @property
    def commands_sent(self) -> int:
        """How many commands this driver has sent the jaws since it was built, the ones a connect sent included.

        One per toggle change and one per solenoid actuation, counted once its commanding write
        (the change, the level, the coil) has been tried: a write that raised may still have
        reached the controller, so the count never says less went out than may have. Never reset,
        not even by a connect: a caller that wants what one call sent reads it before and after.
        The bench's ``--jaws`` reads it so its summary says what went out across the connect and
        the command, and never "nothing was sent" where a connect's question sent a change.
        """
        return self._commands_sent

    @property
    def toggles_without_sensor(self) -> bool:
        """``True`` for a ``single_toggle``: every command is one change that moves the jaws, and nothing reads them."""
        return self._actuation == "single_toggle"

    @property
    def edge_unknown(self) -> bool:
        """``True`` where nobody can say where a toggle's jaws stand: a change whose write failed or never read back,
        or an output somebody switched by hand since the program's last command.

        Until a person says where the jaws stand again (the connect's question, or a pick-start re-ask), every
        change is refused. A caller that must never block on that question, such as a console run's thread,
        reads this and stops instead of asking.

        It reads what the driver knows, never the output: a protocol check reads every data member of the
        protocol (``isinstance`` on Python 3.11), so a property here must not reach the controller. An output
        switched by hand that no command has read yet is :meth:`why_jaws_unknown`'s, which reads it.
        """
        return self._edge_unknown

    def why_jaws_unknown(self) -> str:
        """Why nobody can say where a toggle's jaws stand, read now, without asking or sending anything; ``""`` where
        the count stands.

        A change whose write failed or never read back, or an output somebody switched by hand
        since the program's last command, which this reads off the output. A caller that must
        never block on a question, such as a console run's thread before each pick, asks this and
        stops where it answers anything. A method and not a property, so no protocol check calls it.
        """
        if self._edge_unknown:
            return self._unknown_why or "the last change of its output failed, so nobody can say where they stand"
        return self._switched_clause() if self._connected else ""

    @contextlib.contextmanager
    def asking_nobody(self, why: str) -> Iterator[None]:
        """Within this block, on this thread, nobody is asked where the jaws stand: a question the driver would ask is
        a refusal instead, naming ``why``, and nothing is sent where the answer would have led to a change.

        For a caller that runs where no person can answer, such as a run started from the console:
        its thread must never wait on the server's terminal, and a pick asks the hand at its start
        and again before its approach, seconds apart, where an output switched at the pendant in
        between would otherwise ask (review of 2026-09-28). Other threads ask as before.
        """
        previous = getattr(self._nobody_asked, "why", "")
        self._nobody_asked.why = why or "nobody can answer on this thread"
        try:
            yield
        finally:
            self._nobody_asked.why = previous

    @property
    def asks_at_connect(self) -> bool:
        """Whether :meth:`connect` asks a person where the jaws stand: a ``single_toggle`` always, a solenoid opted in."""
        return self._asks_at_start

    @property
    def where_pin(self) -> str:
        """The bank and pin the jaws are driven on, as a person reads it on the pendant: ``tool output 0``.

        Read off the driver's own configuration, without its lock and without the controller, so it never waits and
        never raises: the console reads it on every poll, also while a question holds the hand for minutes.
        """
        return self._where_pin()

    @property
    def before_change(self) -> "BeforeChange | None":
        """The gate the console installed with its seam (:meth:`answer_questions_with`), or ``None``.

        Read only. Setting it here is refused with ``AttributeError``: the gate comes and goes with the seam it was
        installed for, so a gate set on its own would outlive that seam, or be dropped by the next install without a
        word. Install it with ``answer_questions_with(ask, before_change=gate)``, or hand it to one check,
        ``confirm_where_the_jaws_stand(reason, before_change=gate)``.
        """
        return self._before_change

    @before_change.setter
    def before_change(self, gate: "BeforeChange | None") -> None:
        raise AttributeError(
            "JawIOGripper.before_change is installed with the console's seam, never set on its own: "
            "answer_questions_with(ask, before_change=gate) installs it for every question, and "
            "confirm_where_the_jaws_stand(reason, before_change=gate) asks it at one check")

    # --- who is asked -----------------------------------------------------
    def answer_questions_with(
        self, ask: "StructuredJawQuestion | None", *, before_change: "BeforeChange | None" = None,
    ) -> "StructuredJawQuestion | None":
        """Ask every question where the jaws stand through ``ask`` from now on; the seam it takes the place of.

        ``ask`` gets one :class:`JawAsking` per attempt, stage by stage, and returns one of its ``choices``; an answer
        that is none of them is asked again saying why, ``EOFError`` is no answer, and a question nobody answered is
        refused, never taken for "open". It comes before the ``ask`` handed to the constructor and before the terminal;
        ``None`` takes it away, and they ask again, as a CLI program always did. ``before_change`` comes and goes with
        it: the gate asked immediately before every change a question leads to (:meth:`confirm_where_the_jaws_stand`
        says how). Set under the lock, so it waits for a question in flight.
        """
        with self._lock:
            previous = self._structured
            self._structured = ask
            self._before_change = before_change if ask is not None else None
            return previous

    def question_seam_installed(self) -> bool:
        """Whether a structured seam asks this hand's questions (:meth:`answer_questions_with`).

        The console refuses a connect and a check without one: it never falls back to the terminal of the server it
        runs in, where a question cannot be cancelled and nobody watching the browser sees it.
        """
        return self._structured is not None

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
        :attr:`is_connected` is ``False`` and every command is refused, so no other thread can switch
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
            self._stroke_left_out()
            self._generation = generation = next(self._generations)
            # A connect that raises sets nothing: it never marked the hand connected, so the next one starts again,
            # question included.
            if self._actuation == "single_toggle":
                self._connect_toggle()
            else:
                self._connect_solenoid()
            if self._generation != generation:
                # Past the question, which checks this itself: a disconnect came while the change the person chose, or
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
        still waiting on this one finds that when its answer comes back and sends nothing.
        """
        self._connected = False
        self._generation = next(self._generations)
        # This says what was left behind, because the driver deliberately does not
        # release: a cell found holding a part next morning is explained by this line.
        left = (f"UNKNOWN, {self._unknown_why or 'the count was lost'}" if self._edge_unknown
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
                f"JawIOGripper: a single_toggle takes no width ({float(width_mm)} mm was sent): every change of its "
                "output moves the jaws and nothing measures them. Say open or close with set_closed(), as the hand "
                "verbs do."
            )
        self.set_closed(float(width_mm) <= self._closed_below_mm)

    def set_closed(self, closed: bool, *, wait: bool = True) -> float | None:
        """Close or open the jaws by intent, whatever ``closed_below_mm`` says (``OpensAndCloses``).

        The width a caller sends is read against ``closed_below_mm``, and a grasp above it
        is an open (owner's cell, 2026-09-23). A verb that knows what it means, a pick's
        pre-open, its grasp, a release, says it here instead, so no width can turn it round.
        A close waits for the jaws exactly as a closing width does.

        A ``single_toggle`` changes its output once where its count says the jaws stand the
        other way, leaves it there, and then waits ``close_settle_s``, the stroke, whichever
        way it went. Where they already stand there it sends nothing and says so in the log.
        An output somebody switched since the last command is refused (:meth:`_toggle`).

        ``wait`` false leaves the stroke to the caller where its wait is the travel time alone: the
        change goes out exactly as it would, once, and the moment the stroke is over is returned,
        ``time.monotonic()`` seconds, for :meth:`wait_settled`, so the caller may judge its next
        leg meanwhile and sends nothing until then (``robot.motion.judge_next_leg``, the owner,
        2026-10-09). ``None`` where nothing moved, the jaws already standing there, and where the
        wait reads a switch (a closing solenoid with feedback): that wait runs here, as ever. The
        next command waits out a stroke left so before it changes anything, as it waited inside
        this call before. ``wait`` true, the default, returns ``None`` once the stroke is over.

        It holds the lock, so it waits while a person is being asked where the jaws stand.
        """
        with self._lock:
            self._require_connected("set_closed")
            self._stroke_left_out()
            close = bool(closed)
            if self._actuation == "single_toggle":
                if self._toggle(close):
                    return self._stroke(wait)
                return None
            was_closed = self._closed
            self._actuate(close=close)
            if close:
                if not wait and not self.has_feedback:
                    # Nothing to poll: the travel time is the whole wait, and the caller waits it out.
                    return self._stroke(wait)
                self._await_close()
            elif was_closed and self._open_confirm_pin is None:
                # The jaws travel the same stroke open as closed, and with no open switch the travel time is the only
                # wait there is. Without it a place returned the moment the valve was told, and the arm backed out of
                # a Hand-E that takes up to about 2 s to open, dragging the part it had just set down (owner's cell,
                # 2026-09-23).
                return self._stroke(wait)
            return None

    def wait_settled(self, deadline: float | None) -> None:
        """Wait out what is left of a stroke :meth:`set_closed` left to its caller (``wait=False``): until ``deadline``,
        ``time.monotonic()`` seconds, and at once where it has passed or is ``None``. It sends nothing and takes no lock:
        the change went out with the command, and the next command waits out the stroke itself."""
        if deadline is None:
            return
        left = float(deadline) - time.monotonic()
        if left > 0.0:
            self._sleep(left)
        if self._settles_at is not None and self._settles_at <= float(deadline):
            self._settles_at = None

    def _stroke(self, wait: bool) -> float | None:
        """The jaws' stroke after a change that moved them: waited out here (``wait``), else its end, for the caller."""
        if wait:
            self._sleep(self._close_settle_s)
            return None
        self._settles_at = time.monotonic() + self._close_settle_s
        return self._settles_at

    def _stroke_left_out(self) -> None:
        """Wait out a stroke :meth:`set_closed` left to its caller before anything else changes the jaws: a change while
        they still travel turns them round mid-stroke. Called under the lock, as the stroke was waited there before."""
        settles_at = self._settles_at
        if settles_at is None:
            return
        left = settles_at - time.monotonic()
        if left > 0.0:
            self._sleep(left)
        self._settles_at = None

    def jaws_open_for_a_pick(self) -> str:
        """Before a pick moves the arm: ``""`` where the jaws stand open, else why the pick must not start.

        ``TogglesWithoutSensor``: a pick on a toggle never switches its output before the arm
        moves. Where the count says open, and the output stands where the program left it,
        this answers at once. Where the count says closed (a pick with no place before it) or
        cannot say (a change whose write failed or never read back, or an output somebody
        switched by hand since the last command, the pendant above all), it asks the person
        again, as :meth:`connect` does, and a person who answers closed may choose one change
        to open them now. Here an empty line is no answer: the program already believes
        something else, so only the word ``open`` or ``closed`` is, and an Enter meant for
        something else is asked again. With nobody to ask, the pick is refused. A question
        whose connection changed while it waited is refused without a change. Every other
        actuation answers ``""``: its pick opens the jaws by command.
        """
        with self._lock:
            self._require_connected("jaws_open_for_a_pick")
            if self._actuation != "single_toggle":
                return ""
            if not self._edge_unknown:
                self._switched_by_hand()
            if not (self._closed or self._edge_unknown):
                return ""
            why = (f"a pick starts with them open, and {self._unknown_why}" if self._edge_unknown else
                   "a pick starts with them open, and the program believes they stand CLOSED: its last command "
                   "closed them")
            refused = self._where_the_jaws_stand(why, at_connect=False)
            return f"not starting the pick: {refused}" if refused else ""

    def confirm_where_the_jaws_stand(self, reason: str, *, before_change: "BeforeChange | None" = None) -> str:
        """Ask a person where the jaws stand, whatever the count says; ``""`` where they stand open by the end, else the
        one-line refusal.

        For a check before the arm moves again after a stop (a Restart, Home after a stop, Setup): a person may have
        held the part and opened the jaws at the pendant, or emptied the hand, or not, and a toggle's count says only
        what the program commanded. So this ALWAYS asks, also where the count says open, with the pick start's wording:
        only the word ``open`` or ``closed`` answers it, an empty answer is no answer. ``open`` starts the count there.
        ``closed`` is where the count stands from then on, and the person chooses the one change that opens the jaws or
        an abort; before that change every gate there is is asked, the one :meth:`answer_questions_with` installed and
        ``before_change``, under the lock, and one that answers anything but the empty sentence (no sentence at all
        included), or raises, refuses it with nothing sent. A disconnect that lands while a gate reads the controller
        refuses it too: the connection is read again right before the change. No answer is a refusal, never "open",
        and leaves the count as it was. A caller that reads :attr:`commands_sent` before and after knows whether that
        change went out: a person's hands were at the jaws, and open jaws hold no part. The output switched at the
        pendant while the person decided raises :class:`RobotError` from the change, with nothing sent, and the count
        is nobody's until a person is asked again.

        Where the program's own count says CLOSED and the output still reads the level that count left it at (no change
        since failed or went unread, and nobody switched it at the pendant), "where" is not asked: an answer of ``open``
        there would contradict the program's evidence, turn the count round with nothing sent, and leave a part the
        task closed on clamped while every caller took the hand for empty (review of 2026-10-02: one click on "Offen"
        after a halt in the carry let Restart and Home drive the part, and the next close dropped it). Only what follows
        from closed is asked, ``open_now`` (one change, while a person holds the part) or ``abort``, by the rules above;
        an abort keeps the count CLOSED, and its refusal says how a count starts anew where the jaws stand open all the
        same (connect again: the connect asks where they stand).

        A hand that is not a toggle answers ``""`` at once: it reads its jaws, or opens them by command. Refused
        (``RobotConnectionError``) on a hand that is not connected, and with nobody to ask (a thread under
        :meth:`asking_nobody`, or no seam, no question and no terminal) the answer is the refusal. Holds the lock, so a
        command from another thread waits while the person is asked.
        """
        with self._lock:
            self._require_connected("confirm_where_the_jaws_stand")
            if self._actuation != "single_toggle":
                return ""
            return self._where_the_jaws_stand(reason, at_connect=False, before_change=before_change,
                                              counted=self._counted_closed())

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
            # An interrupt or a coil that never read back must not leave it energised: that is what cooks it. Nor may a
            # halt that lands inside the pulse: a halted arm refuses the plain write and sends nothing, so the coil's
            # end goes out through the arm's own door for it (``end_pulse``: only ever low). Never a toggle's write.
            end_pulse(self._io, pin, port=self._port)

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
        """How long a write is read back before it counts as never shown.

        A solenoid: its pulse and two cycles, 50 ms at least. A toggle: :data:`_CHANGE_READ_BACK_S`, since it
        changes its output once and pulses nothing.
        """
        if self._actuation == "single_toggle":
            return _CHANGE_READ_BACK_S
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
        """A toggle's command: one change of its output where the count says the jaws stand the other way; whether it
        changed.

        Every change moves the jaws whatever they are doing, so a change sent on a wrong count
        moves them the wrong way. So the output is read first: one that no longer stands where
        the program left it was switched by somebody else (:meth:`_switched_by_hand`), and after
        that, or after a change whose write failed or never read back, nobody can say which way
        the next one would move them. This refuses then, before anything is sent, until a person
        has said where the jaws stand; mid-pick that ends the pick, and the next pick asks.
        """
        self.logger.info("actuating jaws %s via single_toggle", "CLOSED" if close else "OPEN")
        if not self._edge_unknown:
            self._switched_by_hand()
        if self._edge_unknown:
            self.logger.warning("not switching %s: %s; nothing was sent", self._where_pin(), self._unknown_why)
            raise RobotError(
                f"JawIOGripper: {self._unknown_why}, so the next change of {self._where_pin()} could move the jaws "
                "either way. Nothing was sent: the next pick asks where they stand, or connect again."
            )
        if self._closed == close:
            self.logger.info("jaws already stand %s: no change", "CLOSED" if close else "OPEN")
            return False
        self._change_toggle()
        return True

    def _change_toggle(self) -> None:
        """One move of the jaws: the output switched once, to the level it does not stand at, and left there.

        Every change of the output moves the jaws, switched on as much as switched off (the
        owner at the pendant, 2026-09-28), so a command is exactly one write, and nothing
        switches it back: the pulse this replaced, low, high and low again, moved them two or
        three times.

        The count flips once the output reads back the new level, not when the write returns:
        a write that returns says the controller accepted it, not that the pin moved
        (:meth:`_reads_back`, review of 2026-09-24). An output that never reads it back within
        :meth:`_read_back_s` flips nothing, leaves ``_edge_unknown`` set and raises, so the next
        pick asks where the jaws stand. A write that raises may still have reached the
        controller with its reply lost, so whether the jaws moved is unknowable:
        ``_edge_unknown`` stays set, and the driver refuses its next change until a person has
        said where the jaws stand.
        """
        pin = self._close_pin
        level = self._read_level() if self._level is None else self._level
        to = not level
        self._edge_unknown = True
        self._unknown_why = (f"the last change of {self._where_pin()} failed on its write, which may still have "
                             "reached the controller, or the output never read its new level back, so nobody can "
                             "say whether the jaws moved")
        self._commands_sent += 1
        self._io.set_digital_output(pin, to, port=self._port)
        if not self._reads_back(pin, to):
            raise RobotError(self._never_read(pin, to, (
                "nobody can say whether the jaws moved. The change is not counted: the next pick asks where they "
                "stand, or connect again")))
        self._closed = not self._closed
        self._level = to
        self._edge_unknown, self._unknown_why = False, ""

    def _read_level(self) -> bool:
        """The level a toggle's output reads now, as the controller reports it."""
        return bool(self._io.get_digital_output(self._close_pin, port=self._port))

    def _switched_by_hand(self) -> bool:
        """Whether a toggle's output no longer stands where the program left it; it then stops trusting its count.

        Every change moves the jaws, so an output switched by somebody else, at the pendant's
        I/O tab where the owner opens the jaws by hand, moved them without the program knowing,
        and a command sent on that count moves them the wrong way. The count is taken as
        unknowable (``_edge_unknown``), and the next pick asks where the jaws stand. Before the
        first answer of a connect nothing is compared.
        """
        clause = self._switched_clause()
        if not clause:
            return False
        self._edge_unknown = True
        self._unknown_why = clause
        self.logger.warning("%s", clause)
        return True

    def _switched_clause(self) -> str:
        """Read only: the clause for a toggle's output that no longer stands where the program left it, else ``""``.

        Nothing is compared before the first answer of a connect. An output that cannot be read
        is taken as one nobody can vouch for, rather than as one that stands where it was left.
        """
        if self._actuation != "single_toggle" or self._level is None:
            return ""
        try:
            now = self._read_level()
        except Exception as exc:  # noqa: BLE001 (a link that cannot say where the output stands vouches for nothing)
            return (f"{self._where_pin()} could not be read ({type(exc).__name__}: {exc}), so nobody can say whether "
                    "it still stands where the program left it")
        if now == self._level:
            return ""
        said = {True: "HIGH", False: "LOW"}
        return (f"{self._where_pin()} reads {said[now]} and the program left it {said[self._level]}: somebody "
                "switched it since the program's last command, at the pendant for instance, so the program's count "
                "no longer says where the jaws stand")

    def _connect_toggle(self) -> None:
        """A toggle's connect: the output is read, never written, and a person says where the jaws stand.

        Nothing is written here: every change of the output moves the jaws, so the low this
        connect used to write on an output left high (by the pendant above all) moved them
        before the question was even asked, and a person then answered about jaws that had just
        moved. The count is never carried over from before, a disconnect or another program,
        because a person or the pendant may have moved the jaws in between; only a person
        looking at them can start it, and it starts only once the answer is in, from the level
        the output reads then. A refusal leaves the gripper disconnected, and ``connect_cell``
        rolls the arm back.
        """
        self._level = None
        self._refuse_unless_open("every change of its output moves them and nothing reads them back, so only a "
                                 "person can say where the program's count of its changes starts")
        self.logger.info("connected on %s: single_toggle on close pin=%s, which reads %s and was not written; the "
                         "jaws stand OPEN, and the program counts its changes from here", self._port.value,
                         self._close_pin, "HIGH" if self._level else "LOW")

    def _refuse_unless_open(self, reason: str) -> None:
        """At connect: ask where the jaws stand, and refuse the connect unless they stand open by the end of it."""
        refused = self._where_the_jaws_stand(reason, at_connect=True)
        if refused:
            raise RobotError(f"JawIOGripper: not connecting: {refused}")

    def _where_the_jaws_stand(self, reason: str, *, at_connect: bool,
                              before_change: "BeforeChange | None" = None, counted: str = "") -> str:
        """Ask a person whether the jaws stand open; ``""`` where they do by the end, else the one-line refusal.

        ``reason`` says why the question is asked, in the question and in a refusal. ``open``:
        they stand open, and the count starts there. ``closed``: that is where the count stands
        from then on, whatever comes next, and the person chooses ``p``, one command now that
        opens them (a single change of a toggle's output, which releases whatever is between
        them), and the stroke is waited out, or ``a``, which aborts. Nothing moves without that
        choice, and before that one command every gate there is is asked, under the lock: the one
        :meth:`answer_questions_with` installed and ``before_change``; one that answers anything but
        the empty sentence, or raises, refuses it with nothing sent. An answer that is none of these is asked again,
        and after :data:`_ASKS` of them, or at end of input, nothing is taken for granted: the
        answer is an abort. For a toggle, the level its output reads once the person has answered
        is where the program's count starts: an output switched after that was switched by hand.

        Enter answers open only ``at_connect``, the owner's agreed answer at program start, and only
        at a terminal or through a string seam: a structured seam offers no default, and its
        question says so, at connect too. At a pick start the program already believes the jaws
        stand closed, or cannot say, so an empty line there, which an Enter meant for something
        else also gives, is no answer: only the word is, and the question says so.

        The answer is acted on only where the connection it was asked on still stands when it
        comes back, and again once every gate has been asked, right before the change: a
        disconnect or another connect meanwhile (another thread, which the lock does not hold off a
        disconnect) makes it an answer about a count nobody keeps any more, and it is refused with
        nothing sent.

        ``counted`` (:meth:`_counted_closed`, a check's alone) is the program's own evidence that the jaws stand
        CLOSED: "where" is not asked then, since an answer of open would contradict it and turn the count round with
        nothing sent; only the second question is, open them now or abort, by the rules above.
        """
        where = self._where_pin()
        nobody = getattr(self._nobody_asked, "why", "")
        if nobody:
            self.logger.warning("nobody is asked where the jaws on %s stand (%s): %s; nothing was sent", where,
                                reason, nobody)
            return f"nobody is asked where the jaws on {where} stand ({reason}): {nobody}"
        seam = self._structured
        ask = None if seam is not None else self._question()
        if seam is None and ask is None:
            return (f"nobody can be asked where the jaws on {where} stand ({reason}): stdin is not a terminal and "
                    "no question was handed to the gripper; run it from a terminal")
        asked_on = (self._generation, self._connected)
        if counted:
            # The program's own count says CLOSED, and the output still stands where that count left it: the program
            # closed them, and nothing has moved them since. "Where" would offer "open" against that evidence: one
            # wrong click turned the count round with nothing sent, the planner was told the hand was empty, and the way
            # back drove the part a task had closed on (review of 2026-10-02). Only what follows from closed is asked.
            self.logger.warning("the jaws on %s are not asked where they stand (%s): %s; only whether to open them now",
                                where, reason, counted)
            said = f"{counted[:1].upper()}{counted[1:]}"
            return self._open_now_or_abort(
                ask, seam, (f"\nThe jaws of the {self._actuation} hand on {where}: {reason}.\n{said}. [p] open them now "
                            f"with {self._how_it_opens()}, which releases anything between them, so hold the part "
                            "first; [a] abort: "),
                reason=f"{reason}; {counted}", at_connect=at_connect, before_change=before_change, asked_on=asked_on,
                closed_by=f"the program's count says the jaws on {where} stand CLOSED, and a person",
                open_after_all=(" If they stand open all the same, connect again: the connect asks where they stand "
                                "and starts the count anew."))
        if at_connect and seam is None:
            question = (f"\nThe jaws of the {self._actuation} hand on {where}: {reason}.\n"
                        "Look at them. Do they stand OPEN? [Enter or 'open' = open, 'closed' = closed]: ")
            choices = {"": "open", "o": "open", "open": "open", "c": "closed", "closed": "closed"}
        else:
            question = (f"\nThe jaws of the {self._actuation} hand on {where}: {reason}.\n"
                        "Look at them. Do they stand OPEN? [type 'open' or 'closed'; Enter alone is no answer here]: ")
            choices = {"o": "open", "open": "open", "c": "closed", "closed": "closed"}
        answer = self._answer(ask, question, choices, stage="where", offered=_WHERE_CHOICES, at_connect=at_connect,
                              reason=reason, seam=seam)
        changed = self._changed_while_asked(asked_on, reason)
        if changed:
            return changed
        if answer == "open":
            self._closed, self._edge_unknown, self._unknown_why = False, False, ""
            self._start_the_count_here()
            self.logger.warning("a person said the jaws on %s stand OPEN (%s)", where, reason)
            return ""
        if answer is None:
            return f"no answer said where the jaws on {where} stand ({reason})"
        # A person looked and said CLOSED: that is the count from here, whatever is chosen next. Kept OPEN, the next
        # close would be a change that opens jaws a person just saw closed (found writing the console's check).
        self._closed, self._edge_unknown, self._unknown_why = True, False, ""
        self._start_the_count_here()
        return self._open_now_or_abort(
            ask, seam, (f"They stand CLOSED. [p] open them now with {self._how_it_opens()}, which releases anything "
                        "between them; [a] abort: "),
            reason=reason, at_connect=at_connect, before_change=before_change, asked_on=asked_on,
            closed_by=f"a person said the jaws on {where} stand CLOSED and")

    def _open_now_or_abort(self, ask: "JawQuestion | None", seam: "StructuredJawQuestion | None", question: str, *,
                           reason: str, at_connect: bool, before_change: "BeforeChange | None",
                           asked_on: "tuple[int, bool]", closed_by: str, open_after_all: str = "") -> str:
        """The second question, the jaws standing CLOSED: ``p``, the one command that opens them, or ``a``, an abort.

        ``""`` once they stand open by the end, else the one-line refusal, nothing sent. ``closed_by`` says who says
        they stand closed, ending where a refusal goes on with what the person did ("a person said the jaws on tool
        output 0 stand CLOSED and", "the program's count says ..., and a person"); ``open_after_all`` is added to an
        abort's or a missing answer's refusal (what to do where they stand open all the same). Before the one command
        every gate is asked (:meth:`_refused_before_the_change`), and the connection the question was asked on is read
        again after the answer and after the gates: a disconnect meanwhile sends nothing.
        """
        where = self._where_pin()
        choice = self._answer(ask, question,
                              {"p": "pulse", "pulse": "pulse", "open_now": "pulse", "a": "abort", "abort": "abort"},
                              stage="open_now", offered=_OPEN_NOW_CHOICES, at_connect=at_connect, reason=reason,
                              seam=seam)
        changed = self._changed_while_asked(asked_on, reason)
        if changed:
            return changed
        if choice != "pulse":
            # An abort is a choice; three wrong answers or the end of input are not, and the refusal says which it was.
            said = "chose not to open them" if choice == "abort" else "gave no clear answer to whether to open them"
            self.logger.warning("%s %s (%s)", closed_by, said, reason)
            return f"{closed_by} {said}" + (f".{open_after_all}" if open_after_all else "")
        refused = self._refused_before_the_change(before_change, where, chose=f"{closed_by} chose to open them")
        if refused:
            return refused
        # A gate reads the controller while the answer stands: a disconnect that landed meanwhile makes it stale too.
        changed = self._changed_while_asked(asked_on, reason, during="the check before the change ran")
        if changed:
            return changed
        self.logger.warning("%s had them opened (%s)", closed_by, reason)
        self._stroke_left_out()
        if self._actuation == "single_toggle":
            self._toggle(False)
        else:
            self._actuate(close=False)
        self._sleep(self._close_settle_s)
        return ""

    def _counted_closed(self) -> str:
        """The program's own evidence that a toggle's jaws stand CLOSED, in a clause, or ``""`` where it has none.

        Its count says CLOSED (its own last change closed them, or a person said so since), no change since failed or
        went unread, and the output still reads the level that count left it at, read now and never written: nothing
        but the program moved the jaws since. Every other hand, a count that says open, a count nobody can vouch for and
        an output that cannot be read have no such evidence.
        """
        if self._actuation != "single_toggle" or not self._closed or self._edge_unknown or self._level is None:
            return ""
        if self._switched_clause():
            return ""
        said = "HIGH" if self._level else "LOW"
        return (f"the program's count says they stand CLOSED, and {self._where_pin()} still reads {said}, as that count "
                "left it, so nothing but the program has moved them since")

    def _refused_before_the_change(self, before_change: "BeforeChange | None", where: str, *,
                                   chose: str = "") -> str:
        """``""`` where every gate asked right before the one change that opens the jaws lets it go out, else the
        one-line refusal, with nothing sent: the gate :meth:`answer_questions_with` installed, then ``before_change``.

        A gate that raises lets nothing through either: a check that could not be made is no check. Only the empty
        sentence lets the change go out: an answer that is no sentence (``None``, ``False``, a number, a blank) is a
        gate that forgot to say, and refuses too. ``chose`` opens the refusal: who says the jaws stand closed, and that
        a person chose to open them.
        """
        chose = chose or f"a person said the jaws on {where} stand CLOSED and chose to open them"
        for gate in (self._before_change, before_change):
            if gate is None:
                continue
            try:
                said = _gate_said(gate())
            except Exception as exc:  # noqa: BLE001 (a check that could not be made lets nothing through)
                said = f"the check before the change could not be made ({type(exc).__name__}: {exc})"
            if said:
                self.logger.warning("not opening the jaws on %s: %s; nothing was sent", where, said)
                return f"{chose}, and the check right before the one change refused it: {said}; nothing was sent"
        return ""

    def _start_the_count_here(self) -> None:
        """A toggle's count starts where a person just said the jaws stand, from the level its output reads now."""
        if self._actuation == "single_toggle":
            self._level = self._read_level()

    def _changed_while_asked(self, asked_on: "tuple[int, bool]", reason: str, *,
                             during: str = "the question waited") -> str:
        """``""`` where the connection a question was asked on still stands, else the refusal that sends nothing.
        ``during`` says what ran while it changed: the question by default, or the check before the change."""
        if (self._generation, self._connected) == asked_on:
            return ""
        where = self._where_pin()
        self.logger.warning("the connection of the hand on %s changed while %s (%s); its answer is not acted on and "
                            "nothing was sent", where, during, reason)
        return (f"the connection of the hand on {where} changed while {during} (a disconnect or another connect came "
                "meanwhile), so its answer is about a count nobody keeps any more; nothing was sent")

    def _answer(self, ask: "JawQuestion | None", question: str, choices: "dict[str, str]", *, stage: str = "where",
                offered: "tuple[str, ...]" = _WHERE_CHOICES, at_connect: bool = False, reason: str = "",
                seam: "StructuredJawQuestion | None" = None) -> "str | None":
        """What the person chose among ``choices``, asked up to :data:`_ASKS` times; ``None`` for no clear answer.

        An answer that is none of the choices, an empty line included where Enter is not one, is
        asked again with the question it answered, prefixed with why. Through a structured
        ``seam`` each attempt is one :class:`JawAsking` of ``stage``, offering ``offered``, and
        only what it offered answers it, or the build plan's own letter for it (``p``, ``a``):
        an empty answer never does, since a browser offers no default, and neither do the
        terminal's ``o``, ``c`` and ``pulse``. ``EOFError`` from either seam is no answer.
        """
        if seam is not None:
            taken = _SEAM_WORDS.get(stage, frozenset(offered))
            choices = {said: meant for said, meant in choices.items() if said in taken}
        asking, why = question, ""
        for attempt in range(1, _ASKS + 1):
            try:
                if seam is not None:
                    said = str(seam(JawAsking(stage=stage, at_connect=at_connect, reason=reason, where=self._where_pin(),
                                              choices=offered, text=asking.strip(), attempt=attempt,
                                              why_again=why))).strip().lower()
                else:
                    assert ask is not None  # the caller asks through the seam or through ask, never neither
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
            return f"one change of {self._where_pin()}"
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
        decided by the execution policy's post-close hold check. Raising would turn one
        that did not catch into a crash.

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
