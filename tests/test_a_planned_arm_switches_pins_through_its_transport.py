"""A planned arm switches I/O only when its vendor lists the I/O mixin, and only through its transport (P0g slice 6).

``SupportsDigitalIO`` decides by structure which gripper a cell may build: a vacuum and an I/O jaw need it, a Robotiq and
an OnRobot do not. Implemented once on ``PlannedArm`` it would make every planned arm claim I/O, so a Franka would be
handed a vacuum it cannot switch. It is ``PlannedDigitalIO``, a mixin a vendor lists when its transport offers
``ControllerIO``, and it holds io1: a write the controller did not take raises instead of returning as if it had.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.robot.core import RobotConnectionError, SupportsDigitalIO
from src.robot.core.arm_capabilities import ArmHalted, DigitalIOPort
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.drivers.planned.capabilities import PlannedDigitalIO
from tests._fake_transport import fake_transport
from tests.test_a_planned_arm_is_built_by_keyword import _pieces


class _SwitchingArm(PlannedDigitalIO, PlannedArm):
    """A vendor whose transport offers I/O, as a Franka's will not."""


def _arm(transport: Any = None) -> Any:
    return _SwitchingArm(**_pieces(transport if transport is not None else fake_transport()))


class OnlyAnArmThatListsTheMixinSwitchesPinsTests(unittest.TestCase):
    def test_a_planned_arm_is_no_io_arm_until_its_vendor_lists_the_mixin(self) -> None:
        self.assertNotIsInstance(PlannedArm(**_pieces(fake_transport())), SupportsDigitalIO)
        self.assertIsInstance(_arm(), SupportsDigitalIO)

    def test_the_ur_arm_lists_it(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        self.assertTrue(issubclass(URRobotArm, PlannedDigitalIO))
        self.assertNotIn("set_digital_output", vars(URRobotArm), "the UR's I/O is the mixin's, not its own")

    def test_a_transport_is_never_mistaken_for_an_io_arm(self) -> None:
        """The transport verbs are named apart from the arm capability's, so a gripper handed a transport by mistake
        is refused by type rather than dropping every False a write answers."""
        from unittest.mock import MagicMock

        from src.robot.drivers.ur.transport import URTransport

        self.assertNotIsInstance(fake_transport(), SupportsDigitalIO)
        self.assertNotIsInstance(URTransport(MagicMock(), MagicMock()), SupportsDigitalIO)


class TheMixinSwitchesThroughTheTransportTests(unittest.TestCase):
    def test_a_write_the_controller_took_reaches_it_with_its_bank(self) -> None:
        transport = fake_transport()
        _arm(transport).set_digital_output(3, True, port=DigitalIOPort.TOOL)
        call = transport.calls("write_digital")[0]
        self.assertEqual((3, True), call.args)
        self.assertIs(DigitalIOPort.TOOL, call.kwargs["port"])
        self.assertTrue(transport.outputs[("tool", 3)])

    def test_a_write_the_controller_did_not_take_raises(self) -> None:
        transport = fake_transport()
        transport.script("write_digital", False)
        transport.script("write_analog", False)
        arm = _arm(transport)
        with self.assertRaisesRegex(RobotConnectionError, "did not take tool digital output 3 = high"):
            arm.set_digital_output(3, True, port=DigitalIOPort.TOOL)
        with self.assertRaisesRegex(RobotConnectionError, "did not take analog output 1 = 2.5 A"):
            arm.set_analog_output(1, 2.5, current=True)

    def test_a_read_answers_the_register_and_the_bank_defaults_to_standard(self) -> None:
        transport = fake_transport()
        transport.inputs[("standard", 2)] = True
        arm = _arm(transport)
        self.assertTrue(arm.get_digital_input(2))
        arm.set_digital_output(5, True)
        self.assertTrue(arm.get_digital_output(5))
        self.assertIs(DigitalIOPort.STANDARD, transport.calls("read_digital_input")[0].kwargs["port"])

    def test_the_end_of_a_pulse_goes_out_low_on_a_halted_arm_and_one_not_taken_raises(self) -> None:
        """The one write a halted arm makes, always low (Willy's own: a bistable coil, a vacuum blow-off)."""
        transport = fake_transport()
        arm = _arm(transport)
        arm.set_digital_output(1, True, port=DigitalIOPort.TOOL)
        transport.request_halt("a person pressed halt")
        with self.assertRaises(ArmHalted):
            arm.set_digital_output(1, False, port=DigitalIOPort.TOOL)
        arm.end_output_pulse(1, port=DigitalIOPort.TOOL)
        self.assertFalse(transport.outputs[("tool", 1)])
        transport.script("end_pulse", False)
        with self.assertRaisesRegex(RobotConnectionError, "did not take tool digital output 1 = low"):
            arm.end_output_pulse(1, port=DigitalIOPort.TOOL)

    def test_what_the_link_raises_propagates(self) -> None:
        transport = fake_transport()
        transport.script("read_digital_input", RuntimeError("receive is down"))
        with self.assertRaisesRegex(RuntimeError, "receive is down"):
            _arm(transport).get_digital_input(0)


if __name__ == "__main__":
    unittest.main()
