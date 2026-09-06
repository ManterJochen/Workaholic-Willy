"""The bench as a noun: the interlock, the bank, and the precedence defect made unconstructible.

⚠ NOTHING HERE TOUCHES HARDWARE and nothing here can. There is no UR controller on a dev box, so the
CLI's interesting paths all end at `connect refused` and cannot be diffed. A fake I/O port is the
only way to exercise what this class actually decides.
"""

from __future__ import annotations

import unittest

from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.drivers.ur.bench import (
    Bench,
    BenchVerdict,
    Measure,
    Pulse,
    Read,
    Set,
    Watch,
)


class FakeIO:
    """A controller's digital I/O. Records every write, including the bank it landed on."""

    def __init__(self, inputs: dict[int, bool] | None = None, deaf: set[int] | None = None) -> None:
        self.inputs = dict(inputs or {})
        self.outputs: dict[int, bool] = {}
        self.writes: list[tuple[int, bool, str]] = []
        self.deaf = set(deaf or ())

    def get_digital_input(self, pin: int, *, port: DigitalIOPort) -> bool:
        return bool(self.inputs.get(pin, False))

    def get_digital_output(self, pin: int, *, port: DigitalIOPort) -> bool:
        return bool(self.outputs.get(pin, False))

    def set_digital_output(self, pin: int, value: bool, *, port: DigitalIOPort) -> None:
        self.writes.append((pin, bool(value), port.value))
        if pin not in self.deaf:
            self.outputs[pin] = bool(value)


def _bench(io: FakeIO, *, allow: bool = True, port: DigitalIOPort = DigitalIOPort.TOOL) -> Bench:
    return Bench.from_arm(io, port=port, confirm=lambda _what: allow)


class InterlockTests(unittest.TestCase):
    """⛔⛔ The only interlock in front of a coil on this path lived in a CLI's main()."""

    def test_a_refused_write_energises_nothing(self) -> None:
        io = FakeIO()
        reading = _bench(io, allow=False).run(Set(pin=4, level=True))
        self.assertIs(reading.verdict, BenchVerdict.REFUSED)
        self.assertEqual(reading.exit_code, 1)
        self.assertEqual(io.writes, [], "the coil was energised despite a refusal")

    def test_every_write_action_asks_and_no_read_does(self) -> None:
        asked: list[str] = []

        def confirm(what: str) -> bool:
            asked.append(what)
            return True

        io = FakeIO(inputs={0: False})
        bench = Bench.from_arm(io, port=DigitalIOPort.TOOL, confirm=confirm)
        bench.run(Read())
        bench.run(Watch(pin=0, for_s=0.01))
        self.assertEqual(asked, [], "a read must never ask permission")

        bench.run(Set(pin=4, level=True))
        bench.run(Pulse(pin=4, for_s=0.001))
        bench.run(Measure(pin=4, level=True, watching=0, for_s=0.01))
        self.assertEqual(len(asked), 3, "every write must ask")
        for sentence in asked:
            self.assertIn("tool", sentence, "the prompt must name the bank being energised")

    def test_confirm_has_no_default(self) -> None:
        """A default returning True is the absence of an interlock wearing its name."""
        import inspect

        for factory in (Bench.from_arm, Bench.from_robot_config):
            param = inspect.signature(factory).parameters["confirm"]
            self.assertIs(param.default, inspect.Parameter.empty, factory.__name__)


class PrecedenceTests(unittest.TestCase):
    def test_read_and_set_cannot_be_asked_for_together(self) -> None:
        """⛔ THE DEFECT, MADE UNCONSTRUCTIBLE. The CLI evaluates five booleans in a fixed order with
        --read first, so `--read --set 4=1 --yes` reads the bank, never writes, and exits 0. An
        action is ONE value here, so "read and also set" is not a sentence you can say.
        """
        import inspect

        params = inspect.signature(Bench.run).parameters
        self.assertEqual(set(params) - {"self"}, {"action"}, "run takes exactly one action")


class BankTests(unittest.TestCase):
    def test_every_write_lands_on_the_benchs_bank(self) -> None:
        """⚠ A pin number without its bank is unusable. Standard output 4 is in the control cabinet;
        tool output 4 is on the wrist."""
        io = FakeIO()
        _bench(io, port=DigitalIOPort.STANDARD).run(Set(pin=4, level=True))
        self.assertEqual([w[2] for w in io.writes], ["standard"])

    def test_the_bank_is_on_the_reading(self) -> None:
        reading = _bench(FakeIO()).run(Set(pin=4, level=True))
        self.assertEqual(reading.port, "tool")
        self.assertIn("tool", reading.render())

    def test_a_config_with_no_io_gripper_refuses_rather_than_guessing(self) -> None:
        """`standard` is what an enum happens to list first, not an answer."""

        class _Gripper:
            vendor = "robotiq"

        class _Cfg:
            gripper = _Gripper()

        with self.assertRaises(ValueError) as caught:
            Bench.from_robot_config(_Cfg(), FakeIO(), confirm=lambda _w: True)
        self.assertIn("no io_port", str(caught.exception))

    def test_the_bank_comes_from_the_configured_gripper(self) -> None:
        class _JawIO:
            io_port = "configurable"

        class _Gripper:
            vendor = "jaw_io"
            jaw_io = _JawIO()

        class _Cfg:
            gripper = _Gripper()

        bench = Bench.from_robot_config(_Cfg(), FakeIO(), confirm=lambda _w: True)
        self.assertIs(bench.port, DigitalIOPort.CONFIGURABLE)


class ReadingTests(unittest.TestCase):
    def test_a_pin_that_does_not_move_is_reported_not_raised(self) -> None:
        """The wrong-bank symptom: the write is accepted and nothing happens. Without the read-back
        it is indistinguishable from success."""
        io = FakeIO(deaf={4})
        reading = _bench(io).run(Set(pin=4, level=True))
        self.assertIs(reading.verdict, BenchVerdict.NO_ANSWER)
        self.assertEqual(reading.exit_code, 2)
        self.assertIn("MISMATCH", reading.render())
        self.assertIn("wrong bank", reading.advice)

    def test_an_input_that_never_answers_is_a_finding_not_an_error(self) -> None:
        reading = _bench(FakeIO(inputs={0: False})).run(
            Measure(pin=4, level=True, watching=0, for_s=0.01)
        )
        self.assertIs(reading.verdict, BenchVerdict.NO_ANSWER)
        self.assertEqual(reading.exit_code, 2, "not 3: this is information an operator acts on")
        self.assertIn("never answered", reading.advice)

    def test_render_is_ascii_without_a_trailing_newline(self) -> None:
        text = _bench(FakeIO(inputs={0: True, 3: True})).run(Read()).render()
        self.assertFalse(text.endswith("\n"))
        text.encode("ascii")

    def test_to_dict_is_json_safe(self) -> None:
        import json

        json.dumps(_bench(FakeIO()).run(Set(pin=4, level=True)).to_dict())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
