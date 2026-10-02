"""A pulse a halt cuts short still ends low: a bistable valve's coil and a vacuum blow-off are never left energised.

A halted arm refuses every output write with ``ArmHalted`` and sends nothing, so the owner's single_toggle Hand-E, whose
every change of tool DO0 moves the jaws, changes nothing (``test_a_halted_arm_sends_nothing_and_switches_no_output``).
Two outputs are different: a double_solenoid valve's coil and a vacuum blow-off are pulsed high and must go back low
however the pulse ends, because a coil held high until the cell is cleared is what cooks it, and a blow-off left on
blows until then. The arm keeps one door open for that, ``end_output_pulse``, which only ever drives low; these hands
go through it when a halt that landed inside their pulse refuses the plain write. Nothing else is switched after the
halt, and an arm that is not halted gets the plain write, as before.

The halted cases run on a real ``URRobotArm`` over ``tests._halt_fakes.SimulatedUr``; the halt lands in the pulse.
"""

from __future__ import annotations

import unittest
from typing import Any, Callable

from src.robot.core import ArmHalted
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.grippers.jaw_io import JawIOGripper
from src.robot.grippers.vacuum import VacuumGripper
from tests._halt_fakes import SimulatedUr, ur_arm_on

_REASON = "the operator pressed halt now"
_TOOL = 16  # SimulatedUr keeps tool output n at 16 + n, as the RTDE receive stream numbers them
_PULSE_S = 0.2


class _HaltInThePulse:
    """The hand's sleep: once armed, the next pulse long enough to be one halts the arm in the middle of it."""

    def __init__(self, arm: Any, sim: SimulatedUr) -> None:
        self.arm, self.sim = arm, sim
        self.armed = False
        self.halted_at: int | None = None

    def __call__(self, seconds: float) -> None:
        if self.armed and self.halted_at is None and seconds > 0.0:
            self.arm.halt(_REASON)
            self.halted_at = len(self.sim.calls)

    def io_since_the_halt(self) -> list[tuple[str, list[Any]]]:
        assert self.halted_at is not None, "the halt never landed inside a pulse"
        return [(call[2], call[3]) for call in self.sim.calls[self.halted_at:] if call[1] == "io"]


def _on_a_ur(make: Callable[[Any, _HaltInThePulse], Any]) -> tuple[Any, SimulatedUr, _HaltInThePulse]:
    sim = SimulatedUr()
    arm = ur_arm_on(sim)
    halt = _HaltInThePulse(arm, sim)
    hand = make(arm, halt)
    hand.connect()
    return hand, sim, halt


class ADoubleSolenoidsCoilTests(unittest.TestCase):
    def _jaws(self, arm: Any, sleep: Callable[[float], None]) -> JawIOGripper:
        return JawIOGripper(arm, actuation="double_solenoid", close_output_pin=0, open_output_pin=1, pulse_s=_PULSE_S,
                            close_settle_s=0.0, min_width_mm=5.0, max_width_mm=49.99, sleep=sleep)

    def test_a_halt_inside_the_close_pulse_still_drops_the_coil_and_switches_nothing_else(self) -> None:
        """Red before: the coil's low write was refused like any other (``ArmHalted``), and the close coil stayed
        energised until the cell was cleared."""
        jaws, sim, halt = _on_a_ur(self._jaws)
        halt.armed = True

        jaws.set_closed(True)

        self.assertFalse(sim.outputs.get(_TOOL + 0, False), "the close coil was left energised")
        self.assertEqual([("setToolDigitalOut", [0, False])], halt.io_since_the_halt())
        self.assertTrue(jaws.jaws_closed, "the pulse ran its length before the halt's write: the valve threw")
        with self.assertRaises(ArmHalted):
            jaws.set_closed(False)
        self.assertEqual([("setToolDigitalOut", [0, False])], halt.io_since_the_halt(),
                         "a new command switched an output on a halted arm")

    def test_a_halt_inside_the_open_pulse_drops_the_open_coil(self) -> None:
        jaws, sim, halt = _on_a_ur(self._jaws)
        jaws.set_closed(True)
        halt.armed = True

        jaws.set_closed(False)

        self.assertFalse(sim.outputs.get(_TOOL + 1, False), "the open coil was left energised")
        self.assertEqual([("setToolDigitalOut", [1, False])], halt.io_since_the_halt())


class AVacuumBlowOffTests(unittest.TestCase):
    def _cup(self, arm: Any, sleep: Callable[[float], None]) -> VacuumGripper:
        return VacuumGripper(arm, vacuum_output_pin=0, blow_off_output_pin=1, io_port=DigitalIOPort.TOOL,
                             blow_off_s=_PULSE_S, sleep=sleep)

    def test_a_halt_inside_the_blow_off_still_ends_it_and_switches_nothing_else(self) -> None:
        """Red before: the blow-off's low write was refused, so it blew until the cell was cleared."""
        cup, sim, halt = _on_a_ur(self._cup)
        cup.set_width_mm(0.0)
        halt.armed = True

        cup.set_width_mm(cup.max_width_mm)

        self.assertFalse(sim.outputs.get(_TOOL + 1, False), "the blow-off was left on")
        self.assertFalse(sim.outputs.get(_TOOL + 0, False), "the vacuum was left on")
        self.assertEqual([("setToolDigitalOut", [1, False])], halt.io_since_the_halt())
        self.assertFalse(cup.vacuum_on)

    def test_a_blow_off_whose_wait_is_interrupted_still_ends_low(self) -> None:
        """Red before: the release wrote the blow-off high, slept, and wrote it low in a straight line, so anything that
        ended the sleep early (an interrupt, a stop of the thread) left it blowing."""
        sim = SimulatedUr()
        arm = ur_arm_on(sim)
        state = {"armed": False}

        def sleep(_seconds: float) -> None:
            if state["armed"]:
                raise KeyboardInterrupt

        cup = self._cup(arm, sleep)
        cup.connect()
        cup.set_width_mm(0.0)
        state["armed"] = True
        with self.assertRaises(KeyboardInterrupt):
            cup.set_width_mm(cup.max_width_mm)
        self.assertFalse(sim.outputs.get(_TOOL + 1, False), "the blow-off was left on")


class _RecordingIO:
    """A digital-I/O source with the arm's end-of-pulse door, recording which of the two each write took."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, int, bool]] = []
        self.levels: dict[int, bool] = {}

    def set_digital_output(self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        self.writes.append(("set", pin, bool(value)))
        self.levels[pin] = bool(value)

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return False

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return self.levels.get(pin, False)

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:  # pragma: no cover
        raise AssertionError("no analog output here")

    def end_output_pulse(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        self.writes.append(("end", pin, False))
        self.levels[pin] = False


class AnArmThatIsNotHaltedTests(unittest.TestCase):
    def test_a_pulse_on_an_arm_that_is_not_halted_ends_with_the_plain_write_as_before(self) -> None:
        io = _RecordingIO()
        jaws = JawIOGripper(io, actuation="double_solenoid", close_output_pin=0, open_output_pin=1, pulse_s=_PULSE_S,
                            close_settle_s=0.0, min_width_mm=5.0, max_width_mm=49.99, sleep=lambda _s: None)
        jaws.connect()
        cup = VacuumGripper(io, vacuum_output_pin=2, blow_off_output_pin=3, blow_off_s=_PULSE_S, sleep=lambda _s: None)
        cup.connect()
        io.writes.clear()

        jaws.set_closed(True)
        cup.set_width_mm(cup.max_width_mm)

        self.assertNotIn("end", [kind for kind, _pin, _level in io.writes], "the halt's door was used unhalted")
        self.assertEqual(("set", 0, False), [w for w in io.writes if w[1] == 0][-1])
        self.assertEqual(("set", 3, False), io.writes[-1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
