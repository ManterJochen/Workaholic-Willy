"""The digital-I/O bench: turn which pin closes the jaws into a measurement.

`confirm` has no default that permits a write, and that is why this class exists. The
only interlock in front of a coil on this path is `_confirm` in the CLI `main()`, gated
by `--yes`. The `io_bench` functions underneath it have no gate at all: `set_output`,
`pulse_output` and `measure_transition` energise a pin the moment they are called.
Publishing a library twin without carrying that seam across would delete the safety
property and call it an API.

It is one verb over an action object rather than five methods, because five methods is
what produces a precedence defect. A CLI that checks five flags in a fixed order with
`--read` first makes ``--read --set 4=1 --yes`` read the bank, never write, and exit 0,
so an operator who typed a write and a read together gets the read in silence. A flag
accepted and ignored answers a question nobody asked. An action is one value here, so
reading and also setting cannot be expressed.

The bank is resolved from the config rather than defaulted. `from_robot_config` reads
`gripper.jaw_io.io_port` or `gripper.vacuum.io_port`, because that is the bank the
gripper of this cell is wired to and measuring that gripper is the point of the bench. A
default here would be a fifth declaration of a fact that already disagreed across four
places.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Sequence

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig
    from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO

__all__ = [
    "Bench",
    "BenchReading",
    "BenchVerdict",
    "Measure",
    "Pulse",
    "Read",
    "Set",
    "Watch",
]


class BenchVerdict(StrEnum):
    """What one bench action established."""

    #: The action ran and answered.
    ANSWERED = "answered"
    #: A write was asked for and the confirmation seam said no. Nothing was energised.
    REFUSED = "refused"
    #: The output was driven and the input never answered, or a watch saw nothing move.
    #: It is not an error: it is the commonest bench result and it is information, from a
    #: wrong pin, a wrong bank, an unpowered sensor or an inverted switch.
    NO_ANSWER = "no_answer"


# --- the actions, each a value rather than a method ---------------------------------------------


@dataclass(frozen=True, slots=True)
class Read:
    """Read every pin on the bank. Read-only, and safe on a live cell at any time."""

    writes: bool = field(default=False, init=False)

    def describe(self, port: str) -> str:
        return f"read every pin on the {port} bank"


@dataclass(frozen=True, slots=True)
class Watch:
    """Poll one input and record every transition. Read-only."""

    pin: int
    #: Not defaulted here. `io_bench.watch_input` and the CLI each declare their own
    #: timeout of 10.0 s, and a third copy in this signature would be a third declaration
    #: of one fact.
    for_s: "Maybe[float]" = UNSET
    writes: bool = field(default=False, init=False)

    def describe(self, port: str) -> str:
        return f"watch input {self.pin} on the {port} bank"


@dataclass(frozen=True, slots=True)
class Set:
    """Drive one output and read it back.

    The read-back is the point. `set_digital_output` returning without raising says the
    command was accepted and not that the pin moved, and a wrong bank looks exactly like
    success.
    """

    pin: int
    level: bool
    writes: bool = field(default=True, init=False)

    def describe(self, port: str) -> str:
        return f"drive output {self.pin} -> {int(self.level)} on the {port} bank"


@dataclass(frozen=True, slots=True)
class Pulse:
    """Drive an output high for a moment, then low. It drops the pin even when interrupted."""

    pin: int
    for_s: "Maybe[float]" = UNSET
    writes: bool = field(default=True, init=False)

    def describe(self, port: str) -> str:
        return f"pulse output {self.pin} on the {port} bank"


@dataclass(frozen=True, slots=True)
class Measure:
    """Drive an output and time how long a watched input takes to answer.

    This is the measurement the whole tool exists for, and the number it returns is the
    floor for `close_settle_s` and `engage_timeout_s`.
    """

    pin: int
    level: bool
    watching: int
    for_s: "Maybe[float]" = UNSET
    writes: bool = field(default=True, init=False)

    def describe(self, port: str) -> str:
        return f"drive output {self.pin} -> {int(self.level)} on the {port} bank"


BenchAction = Read | Watch | Set | Pulse | Measure


@dataclass(frozen=True, slots=True)
class BenchReading:
    """What one action established, on which bank, and what to do about it."""

    verdict: BenchVerdict
    action: BenchAction
    #: The bank is on the reading. A pin number without its bank is unusable, and reading
    #: a level against the wrong bank is the commonest way an afternoon disappears here.
    port: str
    #: The lines a person reads: a snapshot table, a transition log, a timing.
    lines: tuple[str, ...] = ()
    #: What this result means and what to try next.
    advice: str = ""
    #: The timeout used, resolved from the action or from the callee own default. It is
    #: empty for an action that has none.
    timeout_s: float | None = None
    #: For `Measure`: how long the input took, when it answered.
    elapsed_s: float | None = None

    @property
    def exit_code(self) -> int:
        """0 for answered, 1 for refused, 2 for no answer: the three runner codes, derived.

        2 is `_EXIT_TIMEOUT` and is not an error. An output driven with the input never
        answering is a finding an operator acts on, which is why it is separate from 3.
        """
        return {
            BenchVerdict.ANSWERED: 0,
            BenchVerdict.REFUSED: 1,
            BenchVerdict.NO_ANSWER: 2,
        }[self.verdict]

    def render(self) -> str:
        """The reading and what it means. ASCII, no trailing newline, no arguments."""
        head = f"  {self.verdict.value.upper()} on the {self.port} bank"
        out = [head, *(f"  {line}" for line in self.lines)]
        if self.advice:
            out.extend(("", *(f"  {line}" for line in self.advice.splitlines())))
        return "\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "action": type(self.action).__name__.lower(),
            "port": self.port,
            "exit_code": self.exit_code,
            "timeout_s": self.timeout_s,
            "elapsed_s": self.elapsed_s,
            "lines": list(self.lines),
            "advice": self.advice,
        }


@dataclass(frozen=True, slots=True)
class Bench:
    """The digital I/O of a gripper, as something to interrogate deliberately.

        from src.robot.drivers.ur.bench import Bench, Measure

        bench = Bench.from_arm(arm, port=DigitalIOPort.TOOL, confirm=lambda what: input(f"{what}? "))
        reading = bench.run(Measure(pin=4, level=True, watching=0))
        print(reading.render())

    It never moves the arm, and every write goes through `confirm` first.
    """

    io: "SupportsDigitalIO"
    port: "DigitalIOPort"
    #: Required, and able to say no. It is called with a sentence describing exactly what
    #: is about to be energised, and returning False means nothing happens. There is no
    #: default, because a default that returns True is the absence of an interlock
    #: wearing its name.
    confirm: "Callable[[str], bool]"

    # --- two doors -----------------------------------------------------------------------------

    @classmethod
    def from_arm(
        cls, arm: "SupportsDigitalIO", *, port: "DigitalIOPort", confirm: "Callable[[str], bool]"
    ) -> "Bench":
        """An arm that already advertises `SupportsDigitalIO`, on a named bank. The plain-Python door."""
        return cls(io=arm, port=port, confirm=confirm)

    @classmethod
    def from_robot_config(
        cls,
        robot_config: "RobotConfig",
        arm: "SupportsDigitalIO",
        *,
        confirm: "Callable[[str], bool]",
        port: "Maybe[DigitalIOPort]" = UNSET,
    ) -> "Bench":
        """The bank the gripper of this cell is wired to, read from its config.

        It follows the one-builder rule: resolve the bank, then call :meth:`from_arm`.

        It refuses rather than guessing. A config whose gripper is neither `jaw_io` nor
        `vacuum` declares no bank, and the honest answer is that the cell has no
        digital-I/O gripper to exercise, rather than `standard` because that is what an
        enum lists first.
        """
        from src.robot.core.arm_capabilities import DigitalIOPort  # noqa: PLC0415

        if chosen(port):
            return cls.from_arm(arm, port=port, confirm=confirm)
        gripper = getattr(robot_config, "gripper", None)
        vendor = str(getattr(gripper, "vendor", "") or "").lower()
        block = getattr(gripper, vendor, None) if gripper is not None else None
        declared = getattr(block, "io_port", None)
        if declared is None:
            raise ValueError(
                f"robot.gripper.vendor is {vendor!r}, which declares no io_port, so there is no bank "
                f"to exercise. Pass port= explicitly, or configure a digital-I/O gripper "
                f"(jaw_io / vacuum)."
            )
        return cls.from_arm(arm, port=DigitalIOPort(str(declared)), confirm=confirm)

    # --- the verb ------------------------------------------------------------------------------

    def run(self, action: BenchAction) -> BenchReading:
        """Perform one action. A write is confirmed first, and a read never asks.

        It is one action rather than a set of flags. Five booleans evaluated in a fixed
        order make ``--read --set 4=1 --yes`` read the bank, never write and exit 0.
        Here, reading and also setting is not a sentence that can be said.
        """
        from . import io_bench  # noqa: PLC0415

        if action.writes and not self.confirm(action.describe(self.port.value)):
            return BenchReading(
                verdict=BenchVerdict.REFUSED, action=action, port=self.port.value,
                advice="nothing was energised.",
            )

        if isinstance(action, Read):
            snapshot = io_bench.read_snapshot(self.io, port=self.port)
            return BenchReading(
                verdict=BenchVerdict.ANSWERED, action=action, port=self.port.value,
                lines=tuple(snapshot.render().splitlines()),
            )

        if isinstance(action, Set):
            read_back = io_bench.set_output(self.io, action.pin, action.level, port=self.port)
            moved = bool(read_back) == bool(action.level)
            return BenchReading(
                verdict=BenchVerdict.ANSWERED if moved else BenchVerdict.NO_ANSWER,
                action=action, port=self.port.value,
                lines=(
                    f"out {action.pin} := {int(action.level)} -> reads back {int(read_back)}  "
                    f"[{'OK' if moved else 'MISMATCH'}]",
                ),
                advice="" if moved else (
                    "the controller accepted the command and the pin did not move. Usually the "
                    "wrong bank: try standard / configurable / tool."
                ),
            )

        if isinstance(action, Pulse):
            extra: dict[str, Any] = {"seconds": action.for_s} if chosen(action.for_s) else {}
            io_bench.pulse_output(self.io, action.pin, port=self.port, **extra)
            return BenchReading(
                verdict=BenchVerdict.ANSWERED, action=action, port=self.port.value,
                timeout_s=action.for_s if chosen(action.for_s) else None,
                lines=(f"out {action.pin} pulsed (left LOW)",),
            )

        if isinstance(action, Watch):
            watch_extra: dict[str, Any] = (
                {"timeout_s": action.for_s} if chosen(action.for_s) else {}
            )
            log: Sequence[tuple[float, bool]] = io_bench.watch_input(
                self.io, action.pin, port=self.port, **watch_extra
            )
            quiet = len(log) == 1
            return BenchReading(
                verdict=BenchVerdict.NO_ANSWER if quiet else BenchVerdict.ANSWERED,
                action=action, port=self.port.value,
                timeout_s=action.for_s if chosen(action.for_s) else None,
                lines=tuple(f"{at_s:7.3f} s  {int(level)}" for at_s, level in log),
                advice=(
                    "one entry means nothing changed the whole time. Wrong pin, wrong bank, or the "
                    "sensor is unpowered."
                ) if quiet else "",
            )

        measure_extra: dict[str, Any] = (
            {"timeout_s": action.for_s} if chosen(action.for_s) else {}
        )
        result = io_bench.measure_transition(
            self.io, output_pin=action.pin, value=action.level,
            watched_input=action.watching, port=self.port, **measure_extra,
        )
        return BenchReading(
            verdict=BenchVerdict.ANSWERED if result.changed else BenchVerdict.NO_ANSWER,
            action=action, port=self.port.value,
            timeout_s=action.for_s if chosen(action.for_s) else None,
            elapsed_s=float(result.elapsed_s) if result.changed else None,
            lines=tuple(result.render().splitlines()),
            advice=(
                f"use {result.elapsed_s:.3f} s as the FLOOR for close_settle_s / engage_timeout_s. "
                "Repeat it a few times and configure the WORST reading: the driver waits the budget "
                "out, so a typical value drops parts on a slow stroke."
            ) if result.changed else (
                "the output was driven and the input never answered. Check: the right bank, the "
                "right pin, sensor power, and whether the switch is active-LOW."
            ),
        )
