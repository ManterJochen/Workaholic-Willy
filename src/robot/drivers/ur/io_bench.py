"""Bench measurements for a gripper wired to the controller digital I/O.

Every field on ``VacuumGripperConfig`` and ``JawIOGripperConfig`` is a number somebody
measures on the actual cell: which pin closes the jaws, whether the reed switch is
active-high, how long the cylinder takes to travel, how long the ejector needs to build
a seal. That design is what lets the I/O end-effectors be built before the hardware
exists, and it pays off only where those numbers are cheap to obtain. Without a tool,
which pin it is gets answered by editing a YAML, running a pick, watching it fail and
guessing again.

So this is the tool: drive one pin, read one input, and time the transition, with the
arm standing still. Ten bench minutes instead of a debugging session inside a live
pick.

Nothing here commands motion. These functions touch digital I/O and nothing else, and
there is no path from this module to ``arm.move``. That is deliberate, and it is also
what makes the whole thing drivable against an injected I/O port.

Driving an output on a real cell is still a physical action. A close pin closes real
jaws on whatever is between them, and an ejector pin starts real suction. The CLI gates
every write behind an explicit confirmation for that reason: the logic lives here and
the gate lives in ``__main__``.

It is pure by construction. Everything takes a :class:`SupportsDigitalIO` plus injected
``clock`` and ``sleep`` seams, so a caller drives it deterministically and a bench run
gets the real thing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.robot.constants import UR_IO_BENCH_LOG_FILE, create_robot_logger
from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO

__all__ = [
    "IOSnapshot",
    "TransitionResult",
    "measure_transition",
    "pulse_output",
    "read_snapshot",
    "set_output",
    "watch_input",
]

#: The UR standard, configurable and tool banks expose 8 pins at most, and the tool bank
#: has 2.
_PIN_RANGE = range(8)

# A bench session touches real hardware and produces numbers that end up in a YAML file
# weeks later. Logging it lets a config field be traced back to the reading it came
# from, and puts a pin somebody drove on record even once the terminal that printed it
# is gone. The polling loops below stay silent, and only the commanded action and its
# verdict are logged.
logger = create_robot_logger("URIOBench", UR_IO_BENCH_LOG_FILE)


@dataclass(frozen=True, slots=True)
class IOSnapshot:
    """Every readable pin on one bank, at one instant."""

    port: DigitalIOPort
    inputs: dict[int, bool] = field(default_factory=dict)
    outputs: dict[int, bool] = field(default_factory=dict)

    def render(self) -> str:
        """A table to read off the bench, aligned so a changed bit is obvious."""
        # The full probed range rather than the pins that answered. A pin missing from
        # both maps was probed and is not exposed, and dropping it from the table would
        # make a bank with no pin 6 look identical to a pin nobody looked for, which is
        # the confusion this tool exists to remove.
        pins = sorted(set(_PIN_RANGE) | set(self.inputs) | set(self.outputs))
        head = f"port={self.port.value:<13} " + " ".join(f"{p:>3}" for p in pins)
        ins = "  in " + " " * 11 + " ".join(_bit(self.inputs.get(p)) for p in pins)
        outs = "  out" + " " * 11 + " ".join(_bit(self.outputs.get(p)) for p in pins)
        return "\n".join((head, ins, outs))


#: `port` is required on every function below and has no default, because a default is
#: the only thing in this package that could move hardware by being merely plausible.
#: Measured, the same fact was declared in four places and they disagreed:
#:
#:     the five io_bench functions          DigitalIOPort.standard
#:     drivers/ur/__main__.py:119 --port    "tool"
#:     robot_schema.py:206 jaw_io.io_port   "tool"
#:     robot_schema.py:259 vacuum.io_port   "tool"
#:
#: So a Python caller writing the obvious twin of `--set 4=1 --yes` energises a
#: different physical pin from the one the CLI drives and the one the operator
#: configured: standard output 4 is in the control cabinet and tool output 4 is on the
#: wrist. Nothing raises, nothing is logged, and both values are legal members of the
#: same enum.
#:
#: It is required rather than re-defaulted to "tool". A bench exists to drive a pin
#: deliberately and there is no bank it may guess, and guessing the config value would
#: move the fourth declaration rather than delete it. A caller who omits it gets a
#: `TypeError` at the call, which is the fail-closed direction. The cell-aware door that
#: resolves the bank from config is a separate noun, and the vendor Protocol in
#: `core/arm_capabilities.py` keeps its `STANDARD` default because standard genuinely is
#: the controller own default bank, which is a statement about the hardware rather than
#: about this cell.

def _bit(value: bool | None) -> str:
    return "  -" if value is None else ("  1" if value else "  0")


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """What a commanded output did to a watched input, and how long it took.

    ``elapsed_s`` is the number that goes into ``close_settle_s`` and
    ``engage_timeout_s``. Measure it a few times and take the worst rather than the
    best, because the config field is a budget and not a typical value.
    """

    pin: int
    watched_input: int
    #: The input level before the output was driven.
    before: bool
    #: The input level when the wait ended, whether because it changed or because time
    #: ran out.
    after: bool
    elapsed_s: float
    changed: bool
    timed_out: bool

    def render(self) -> str:
        verdict = (
            f"changed {int(self.before)} -> {int(self.after)} after {self.elapsed_s * 1000:.0f} ms"
            if self.changed
            else f"NO CHANGE (stayed {int(self.before)}) after {self.elapsed_s * 1000:.0f} ms"
        )
        note = "  [TIMED OUT]" if self.timed_out else ""
        return f"out {self.pin} -> in {self.watched_input}: {verdict}{note}"


def read_snapshot(io: SupportsDigitalIO, *, port: DigitalIOPort) -> IOSnapshot:
    """Read every pin on one bank. It is read-only and safe on a live cell at any time.

    A pin the controller does not expose raises rather than returning a level, so it is
    recorded as absent with a ``-`` instead of as a confident ``0``. A missing pin and a
    low pin are different facts, and not confusing them is the point of this tool.
    """
    inputs: dict[int, bool] = {}
    outputs: dict[int, bool] = {}
    for pin in _PIN_RANGE:
        try:
            inputs[pin] = bool(io.get_digital_input(pin, port=port))
        except Exception:  # noqa: BLE001 (an unexposed pin is a fact, not a failure)
            pass
        try:
            outputs[pin] = bool(io.get_digital_output(pin, port=port))
        except Exception:  # noqa: BLE001
            pass
    logger.info(
        "read %s bank: %d input(s), %d output(s) answered of %d pins probed",
        port.value, len(inputs), len(outputs), len(_PIN_RANGE),
    )
    return IOSnapshot(port=port, inputs=inputs, outputs=outputs)


def set_output(
    io: SupportsDigitalIO,
    pin: int,
    value: bool,
    *,
    port: DigitalIOPort,
    settle_timeout_s: float = 0.5,
    poll_s: float = 0.01,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> bool:
    """Drive one output and read it back, returning the read-back level.

    The read-back is the point. ``set_digital_output`` returning without raising says
    the command was accepted and not that the pin moved: a wrong bank, a pin the
    controller reserves, and a mix-up between the configurable and standard banks all
    look like success from the caller side. Comparing the read-back against the request
    is what turns a sent command into a pin that is high.
    """
    io.set_digital_output(int(pin), bool(value), port=port)
    # Settle before reading, because the read-back is the whole point and is wrong
    # without it. The output goes out over RTDE and the state comes back on the RTDE
    # receive stream, so an immediate read returns the previous level. Measured against
    # URSim 5.26.0: setting high and reading at once reports False, and the next read,
    # of the following low command, reports True. That one-step lag reads exactly like
    # an inverted pin and would send somebody to check wiring that was fine.
    #
    # It polls rather than sleeping a fixed time: a pin that answers in 5 ms should not
    # cost 200, and a pin that never answers must still return its real level rather
    # than a guess. A settle that times out returns the last real read, so a command
    # accepted with the pin not moving stays reportable, which is the wrong-bank signal
    # this function exists to surface.
    deadline = clock() + float(settle_timeout_s)
    read = bool(io.get_digital_output(int(pin), port=port))
    while read != bool(value) and clock() < deadline:
        sleep(float(poll_s))
        read = bool(io.get_digital_output(int(pin), port=port))
    # No elapsed time here on purpose: reading the clock once more would consume an
    # extra tick from a caller-injected ``clock`` seam, and the contract of this function
    # is that the seams are driven exactly as often as the algorithm needs them.
    # ``measure_transition`` is where timing is measured.
    if read == bool(value):
        logger.info("drove %s output %d := %d, read back %d", port.value, int(pin), int(bool(value)), int(read))
    else:
        # The read-back disagreeing with the command is the wrong-bank signal this
        # function exists to surface, and it is returned rather than raised, so the log
        # is the only durable record.
        logger.warning(
            "drove %s output %d := %d but it reads back %d; wrong bank, a reserved pin, or nothing "
            "wired there",
            port.value, int(pin), int(bool(value)), int(read),
        )
    return read


def pulse_output(
    io: SupportsDigitalIO,
    pin: int,
    *,
    seconds: float = 0.2,
    port: DigitalIOPort,
    sleep: Callable[[float], Any] = time.sleep,
) -> None:
    """Drive an output high for ``seconds``, then low: the double-solenoid and blow-off shape.

    It always drops the pin, including where the sleep is interrupted. A latching coil
    left energised is the failure this exists to avoid, and an interrupt mid-pulse is
    exactly when it would happen.
    """
    logger.info("pulsing %s output %d high for %.3f s", port.value, int(pin), float(seconds))
    io.set_digital_output(int(pin), True, port=port)
    try:
        sleep(float(seconds))
    finally:
        io.set_digital_output(int(pin), False, port=port)


def watch_input(
    io: SupportsDigitalIO,
    pin: int,
    *,
    timeout_s: float = 10.0,
    port: DigitalIOPort,
    poll_s: float = 0.01,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> list[tuple[float, bool]]:
    """Poll one input and record every transition as ``(seconds_since_start, level)``.

    The bench use is to start watching and then trip the sensor by hand, pushing the
    jaws shut or blocking the vacuum port, and read off whether it is active-high,
    whether it bounces and how fast it responds. The first entry is always the starting
    level, so a run that records exactly one entry means nothing moved.
    """
    start = clock()
    level = bool(io.get_digital_input(int(pin), port=port))
    log: list[tuple[float, bool]] = [(0.0, level)]
    while clock() - start < float(timeout_s):
        sleep(float(poll_s))
        now = bool(io.get_digital_input(int(pin), port=port))
        if now != level:
            level = now
            log.append((clock() - start, level))
    # One line for the whole watch rather than one per transition: the transitions are
    # the caller return value, and a bouncing switch would otherwise flood the file.
    logger.info(
        "watched %s input %d for %.1f s: %d transition(s) after the initial %d",
        port.value, int(pin), float(timeout_s), len(log) - 1, int(log[0][1]),
    )
    return log


def measure_transition(
    io: SupportsDigitalIO,
    *,
    output_pin: int,
    value: bool,
    watched_input: int,
    timeout_s: float = 2.0,
    port: DigitalIOPort,
    poll_s: float = 0.005,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> TransitionResult:
    """Drive an output and time how long the watched input takes to respond.

    This is the bench measurement, and it is where ``close_settle_s`` and
    ``engage_timeout_s`` come from: command the close coil, watch the closed-confirm
    reed, and the elapsed time is the travel of the cylinder. Run it several times and
    configure the worst, because the config field is a budget the driver waits out
    rather than an average.

    A timeout is reported and never raised. An output driven with the input never
    answering is a real and common bench result, from a wrong pin, a wrong bank, an
    unpowered sensor or an inverted switch, and it is information rather than an error.
    """
    before = bool(io.get_digital_input(int(watched_input), port=port))
    start = clock()
    io.set_digital_output(int(output_pin), bool(value), port=port)
    while True:
        now = bool(io.get_digital_input(int(watched_input), port=port))
        elapsed = clock() - start
        if now != before:
            # The bench number, which becomes close_settle_s and engage_timeout_s.
            logger.info(
                "%s out %d := %d -> in %d changed %d -> %d after %.0f ms",
                port.value, int(output_pin), int(bool(value)), int(watched_input),
                int(before), int(now), elapsed * 1000.0,
            )
            return TransitionResult(
                pin=int(output_pin), watched_input=int(watched_input), before=before,
                after=now, elapsed_s=elapsed, changed=True, timed_out=False,
            )
        if elapsed >= float(timeout_s):
            logger.warning(
                "%s out %d := %d -> in %d never answered within %.1f s (stayed %d)",
                port.value, int(output_pin), int(bool(value)), int(watched_input),
                float(timeout_s), int(before),
            )
            return TransitionResult(
                pin=int(output_pin), watched_input=int(watched_input), before=before,
                after=now, elapsed_s=elapsed, changed=False, timed_out=True,
            )
        sleep(float(poll_s))
