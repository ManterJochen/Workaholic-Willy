"""The bench exerciser: turn "which pin is it?" into a measurement instead of a guessing loop.

Every wiring number the two digital-I/O grippers need — which pin, which bank, active-high or -low,
how long the cylinder takes — is config on purpose, so the drivers could be written before the
hardware was chosen. That trade only pays off if the numbers are cheap to obtain. Without a tool the
answer to "which pin closes the jaws?" is: edit a YAML, run a pick, watch it fail, guess again — with
a powered arm in the room.

Two behaviours here are load-bearing and both come from the same idea, that a command is not a
measurement:

* ``set_output`` READS THE PIN BACK. ``set_digital_output`` returning without raising says the
  command was accepted, not that the pin moved — a wrong bank looks identical to success.
* ``measure_transition`` REPORTS a timeout instead of raising. "The output was driven and the input
  never answered" is the most common bench result and it is information, not an error.

STATUS: bucket (3) — ur_rtde==1.6.5 IS installed here since 2026-08-17 (and all 33 symbols this repo
calls were verified present), but none of this has spoken to a controller yet. These tests prove the
LOGIC; the wiring is what the tool exists to measure.
"""

from __future__ import annotations

import unittest

from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.drivers.ur import io_bench


class FakeIO:
    """A controller's digital I/O. ``unexposed`` pins raise, as a real controller does."""

    def __init__(self, inputs=None, unexposed=()) -> None:
        self.inputs = dict(inputs or {})
        self.outputs: dict[int, bool] = {}
        self.writes: list[tuple[int, bool, str]] = []
        self.unexposed = set(unexposed)
        #: Set to a pin to make writes silently not take effect (the wrong-bank symptom).
        self.deaf_pins: set[int] = set()

    def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
        self.writes.append((pin, bool(value), str(port)))
        if pin not in self.deaf_pins:
            self.outputs[pin] = bool(value)

    def get_digital_input(self, pin, *, port=DigitalIOPort.STANDARD) -> bool:
        if pin in self.unexposed:
            raise RuntimeError(f"pin {pin} is not exposed")
        return bool(self.inputs.get(pin, False))

    def get_digital_output(self, pin, *, port=DigitalIOPort.STANDARD) -> bool:
        if pin in self.unexposed:
            raise RuntimeError(f"pin {pin} is not exposed")
        return bool(self.outputs.get(pin, False))

    def set_analog_output(self, pin, value, *, current: bool = False) -> None:
        pass


class SnapshotTests(unittest.TestCase):
    def test_it_reads_the_whole_bank_without_writing_anything(self) -> None:
        io = FakeIO(inputs={0: True, 3: True})
        snap = io_bench.read_snapshot(io, port=DigitalIOPort.TOOL)
        self.assertEqual(io.writes, [])                 # read-only: safe on a live cell
        self.assertTrue(snap.inputs[0])
        self.assertFalse(snap.inputs[1])

    def test_an_unexposed_pin_is_absent_not_low(self) -> None:
        # A missing pin and a low pin are different facts, and confusing them is exactly the mistake
        # this tool exists to prevent -- "pin 6 reads 0" would send someone hunting a dead sensor.
        io = FakeIO(inputs={0: True}, unexposed={6, 7})
        snap = io_bench.read_snapshot(io, port=DigitalIOPort.TOOL)
        self.assertNotIn(6, snap.inputs)
        self.assertIn("  -", snap.render())


class SetOutputTests(unittest.TestCase):
    def test_it_reads_the_pin_back(self) -> None:
        io = FakeIO()
        self.assertTrue(io_bench.set_output(io, 4, True, port=DigitalIOPort.TOOL))

    def test_a_command_that_does_not_move_the_pin_is_visible(self) -> None:
        # The wrong-bank symptom: the write is accepted and nothing happens. Without the read-back
        # this is indistinguishable from success, which is how an afternoon disappears.
        io = FakeIO()
        io.deaf_pins = {4}
        self.assertFalse(io_bench.set_output(io, 4, True, port=DigitalIOPort.TOOL))


class PulseTests(unittest.TestCase):
    def test_it_drives_high_then_low(self) -> None:
        io = FakeIO()
        io_bench.pulse_output(io, 4, seconds=0.05, sleep=lambda _s: None, port=DigitalIOPort.TOOL)
        self.assertEqual([(p, v) for p, v, _ in io.writes], [(4, True), (4, False)])

    def test_the_pin_is_dropped_even_if_the_wait_is_interrupted(self) -> None:
        # A latching coil left energised is the failure this guards against, and a Ctrl-C mid-pulse
        # is precisely when it would happen.
        io = FakeIO()

        def boom(_s):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            io_bench.pulse_output(io, 4, seconds=0.05, sleep=boom, port=DigitalIOPort.TOOL)
        self.assertIs(io.outputs[4], False)


class WatchTests(unittest.TestCase):
    def test_it_records_the_starting_level_and_every_transition(self) -> None:
        io = FakeIO(inputs={0: False})
        ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 9.9])
        levels = iter([False, True, True, False])

        def fake_input(pin, *, port=DigitalIOPort.STANDARD):
            try:
                return next(levels)
            except StopIteration:
                return False

        io.get_digital_input = fake_input  # type: ignore[method-assign]
        log = io_bench.watch_input(
            io, 0, timeout_s=0.35, clock=lambda: next(ticks), sleep=lambda _s: None,
            port=DigitalIOPort.TOOL,
        )
        self.assertEqual(log[0][1], False)               # starting level always recorded
        self.assertIn(True, [lvl for _t, lvl in log])    # the transition was caught

    def test_a_quiet_watch_records_exactly_one_entry(self) -> None:
        # One entry is the tool's way of saying "nothing moved" -- wrong pin, wrong bank, no power.
        ticks = iter([0.0, 0.0, 0.5, 1.5])
        log = io_bench.watch_input(
            FakeIO(inputs={0: False}), 0, timeout_s=1.0,
            clock=lambda: next(ticks), sleep=lambda _s: None, port=DigitalIOPort.TOOL,
        )
        self.assertEqual(len(log), 1)


class MeasureTransitionTests(unittest.TestCase):
    """The measurement that produces close_settle_s / engage_timeout_s."""

    def test_it_times_the_input_answering_the_output(self) -> None:
        io = FakeIO(inputs={0: False})
        ticks = iter([0.0, 0.02, 0.05, 0.05])
        state = {"n": 0}

        def fake_input(pin, *, port=DigitalIOPort.STANDARD):
            state["n"] += 1
            return state["n"] > 2                        # answers on the third read

        io.get_digital_input = fake_input  # type: ignore[method-assign]
        res = io_bench.measure_transition(
            io, output_pin=4, value=True, watched_input=0, timeout_s=1.0,
            clock=lambda: next(ticks), sleep=lambda _s: None, port=DigitalIOPort.TOOL,
        )
        self.assertTrue(res.changed)
        self.assertFalse(res.timed_out)
        self.assertGreater(res.elapsed_s, 0.0)
        self.assertIn("changed", res.render())

    def test_an_input_that_never_answers_is_REPORTED_not_raised(self) -> None:
        # The commonest bench result: wrong pin, wrong bank, unpowered sensor, inverted switch. All
        # information. Raising here would turn the tool's most useful output into a stack trace.
        ticks = iter([0.0, 0.5, 1.5, 1.5])
        res = io_bench.measure_transition(
            FakeIO(inputs={0: False}), output_pin=4, value=True, watched_input=0, timeout_s=1.0,
            clock=lambda: next(ticks), sleep=lambda _s: None, port=DigitalIOPort.TOOL,
        )
        self.assertFalse(res.changed)
        self.assertTrue(res.timed_out)
        self.assertIn("NO CHANGE", res.render())

    def test_the_output_is_actually_driven(self) -> None:
        io = FakeIO(inputs={0: False})
        ticks = iter([0.0, 0.5, 1.5, 1.5])
        io_bench.measure_transition(
            io, output_pin=4, value=True, watched_input=0, timeout_s=1.0,
            clock=lambda: next(ticks), sleep=lambda _s: None, port=DigitalIOPort.TOOL,
        )
        self.assertEqual(io.writes[0][:2], (4, True))


class CliArgumentTests(unittest.TestCase):
    """The CLI's refusals, which are the part that keeps a bench session safe."""

    def test_pin_value_parsing_accepts_the_obvious_spellings(self) -> None:
        from backend.src.robot.drivers.ur.__main__ import _parse_pin_value

        for spec, want in (("4=1", (4, True)), ("0=low", (0, False)), ("2=ON", (2, True))):
            self.assertEqual(_parse_pin_value(spec), want)

    def test_an_ambiguous_pin_value_is_refused_rather_than_guessed(self) -> None:
        from backend.src.robot.drivers.ur.__main__ import _parse_pin_value

        for spec in ("4", "4=maybe", "x=1"):
            with self.subTest(spec=spec), self.assertRaises(SystemExit):
                _parse_pin_value(spec)

    def test_measure_without_a_watched_input_is_refused(self) -> None:
        from backend.src.robot.drivers.ur.__main__ import main

        with self.assertRaises(SystemExit):
            main(["--measure", "4=1"])

    def test_doing_nothing_is_refused(self) -> None:
        from backend.src.robot.drivers.ur.__main__ import main

        with self.assertRaises(SystemExit):
            main([])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
