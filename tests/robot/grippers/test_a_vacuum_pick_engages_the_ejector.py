"""A suction pick switches the vacuum on and a place switches it off, whatever width the verb carries (P0g).

Measured 2026-09-24 (the P0g grippers map, with this file's doubles): ``Robot.pick`` of a 40 mm part commanded 39.0 mm,
which a cup reads against ``vacuum_on_below_mm`` as a release. With no vacuum switch wired the pick reported EXECUTED and
a grasp while every write on the ejector pin was ``(0, False)``: the vacuum never came on. The pick loop's close had the
same trap. A cup has no opening, so the hand verbs and the pick loop now tell it the intent (``OpensAndCloses``), as
they tell a jaw_io hand, and a width is only what a direct caller of ``set_width_mm`` means by it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.core.gripper import OpensAndCloses
from src.robot.execution.robot import Robot
from src.robot.grippers.sim.suction_gripper import IsaacSuctionGripper
from src.robot.grippers.vacuum import VacuumGripper
from tests.robot.safety.test_robot_pick_and_place import _Log, _pose, _RecordingArm

_EJECTOR, _BLOW_OFF, _SWITCH = 0, 1, 2


class _IO:
    """The controller's digital I/O: every write recorded, the vacuum switch following the ejector when wired."""

    def __init__(self, *, switch_follows_the_ejector: bool = False) -> None:
        self.writes: list[tuple[int, bool, DigitalIOPort]] = []
        self._follows = switch_follows_the_ejector
        self._ejector = False

    def set_digital_output(self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        self.writes.append((int(pin), bool(value), port))
        if pin == _EJECTOR:
            self._ejector = bool(value)

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return self._follows and pin == _SWITCH and self._ejector

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return pin == _EJECTOR and self._ejector

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        pass

    def ejector(self) -> list[bool]:
        return [value for pin, value, _port in self.writes if pin == _EJECTOR]


def _cup(io: _IO, *, switch: bool = False) -> VacuumGripper:
    cup = VacuumGripper(io, vacuum_output_pin=_EJECTOR, blow_off_output_pin=_BLOW_OFF,
                        vacuum_ok_input_pin=_SWITCH if switch else None, io_port="tool", sleep=lambda _s: None)
    cup.connect()
    return cup


def _robot(cup: VacuumGripper) -> Robot:
    arm = _RecordingArm(_Log())
    arm.connect()
    return Robot.from_parts(arm=arm, gripper=cup, lock_key=None)


class APickEngagesTheVacuumTests(unittest.TestCase):

    def test_a_pick_of_a_part_wider_than_the_threshold_switches_the_vacuum_on(self) -> None:
        io = _IO()
        report = _robot(_cup(io)).pick(_pose(), 40.0)
        self.assertTrue(report.ok, report.render())
        self.assertIn(True, io.ejector(), "the ejector was never switched on")
        self.assertTrue(io.ejector()[-1], "the pick ended with the vacuum off")

    def test_a_place_switches_it_off_and_blows_the_part_off(self) -> None:
        io = _IO()
        robot = _robot(_cup(io))
        robot.pick(_pose(), 40.0)
        io.writes.clear()
        robot.place(_pose(), standoff_mm=50.0)
        self.assertFalse(io.ejector()[-1], "the place left the vacuum on")
        self.assertIn((_BLOW_OFF, True, DigitalIOPort.TOOL), io.writes)

    def test_a_switch_that_confirms_a_seal_is_a_hold(self) -> None:
        io = _IO(switch_follows_the_ejector=True)
        cup = _cup(io, switch=True)
        _robot(cup).pick(_pose(), 40.0)
        self.assertTrue(cup.is_object_detected())


class TheCupTakesOpenAndCloseByIntentTests(unittest.TestCase):

    def test_the_real_cup_is_one(self) -> None:
        io = _IO(switch_follows_the_ejector=True)
        cup = _cup(io, switch=True)
        self.assertIsInstance(cup, OpensAndCloses)
        cup.set_closed(True)
        self.assertTrue(cup.vacuum_on)
        cup.set_closed(False)
        self.assertFalse(cup.vacuum_on)
        self.assertIn((_BLOW_OFF, True, DigitalIOPort.TOOL), io.writes)

    def test_a_width_still_means_what_it_always_meant_to_a_direct_caller(self) -> None:
        """The control: set_width_mm is unchanged, only the verbs that know their intent stopped sending a width."""
        io = _IO()
        cup = _cup(io)
        cup.set_width_mm(40.0)
        self.assertFalse(cup.vacuum_on)
        cup.set_width_mm(0.0)
        self.assertTrue(cup.vacuum_on)

    def test_the_sim_cup_is_one_too(self) -> None:
        cup = IsaacSuctionGripper(session=object(), gripper_prim_path=None, mock_mode=True)
        cup.connect()
        self.assertIsInstance(cup, OpensAndCloses)
        cup.set_closed(True)
        self.assertEqual(0.0, cup.get_width_mm())
        cup.set_closed(False)
        self.assertEqual(cup.max_width_mm, cup.get_width_mm())


class TheBenchJawsVerbStaysAJawVerbTests(unittest.TestCase):
    """``--jaws`` drives a jaw_io hand through its driver under a sentence naming its actuation. A cup that now takes
    open and close by intent must not pass that gate: its confirmation would name a jaw_io actuation it does not have,
    and the command would switch the ejector."""

    def test_a_vacuum_profile_is_refused_before_anything_is_written(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.io_bench import __main__ as bench_cli

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"ip": "192.168.1.100"},
            "gripper": {"vendor": "vacuum", "vacuum": {"vacuum_output_pin": _EJECTOR, "io_port": "tool"}},
        })
        io = _IO()
        arm: Any = io
        arm.connect = lambda: None
        arm.disconnect = lambda: None
        arm.capabilities = SimpleNamespace(vendor="ur", controller="192.168.1.100", owns_a_controller=True)
        asked: list[str] = []
        original = bench_cli._confirm
        bench_cli._confirm = lambda _args, what: bool(asked.append(what)) or True  # type: ignore[assignment]
        try:
            code = bench_cli._drive_jaws(SimpleNamespace(jaws="closed", yes=True, profile=None), cfg, arm)
        finally:
            bench_cli._confirm = original  # type: ignore[assignment]
        self.assertEqual(1, code)
        self.assertEqual([], asked, "nothing was offered for confirmation")
        self.assertEqual([], io.writes, "nothing was written")


if __name__ == "__main__":
    unittest.main()
