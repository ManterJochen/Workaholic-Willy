"""A planned arm reads force, status and provenance and releases a stop only through its transport (P0g slice 7).

The UR driver read these off its RTDE connection itself, so no other vendor could share a line of them, and the planning
half's refusal text read the status through a method only the UR had (under ``type: ignore``). Each is now a transport
Protocol and a mixin a vendor lists when its transport offers it; a planned arm whose transport offers none claims none
of the capabilities, and its refusals carry no state it cannot read.
"""

from __future__ import annotations

import logging
import unittest
from typing import Any

from src.geometry import Frame
from src.robot.core import (
    ArmFault,
    ArmFaultClass,
    ControllerProvenance,
    ReportsControllerProvenance,
    RobotMode,
    RobotStatus,
    SafetyMode,
    SupportsForceTorque,
    SupportsRobotStatus,
    Wrench,
)
from src.robot.core.transport import ArmTransport
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.drivers.planned.capabilities import (
    PlannedControllerProvenance,
    PlannedFaultAcknowledge,
    PlannedForceTorque,
    PlannedRobotStatus,
)
from tests._fake_transport import fake_transport
from tests.robot.drivers.planned.test_a_planned_arm_is_built_by_keyword import _pieces

_STOPPED = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP, protective_stopped=True,
                       emergency_stopped=False, message="PROTECTIVE_STOP", program_running=False)


class _ReportingArm(PlannedForceTorque, PlannedRobotStatus, PlannedFaultAcknowledge, PlannedControllerProvenance,
                    PlannedArm):
    """A vendor whose transport reads force, status and provenance and releases a stop."""


def _arm(transport: Any) -> Any:
    return _ReportingArm(**_pieces(transport))


class OnlyAnArmThatListsTheMixinsReportsTests(unittest.TestCase):
    def test_a_planned_arm_claims_none_of_them_until_its_vendor_lists_them(self) -> None:
        bare = PlannedArm(**_pieces(fake_transport()))
        listed = _arm(fake_transport())
        for capability in (SupportsForceTorque, SupportsRobotStatus, ReportsControllerProvenance):
            with self.subTest(capability=capability.__name__):
                self.assertNotIsInstance(bare, capability)
                self.assertIsInstance(listed, capability)

    def test_status_is_two_mixins_so_an_arm_that_cannot_release_is_no_status_arm(self) -> None:
        """``SupportsRobotStatus`` asks for the status and the acknowledge together."""

        class _StatusOnly(PlannedRobotStatus, PlannedArm):
            pass

        self.assertNotIsInstance(_StatusOnly(**_pieces(fake_transport())), SupportsRobotStatus)

    def test_the_ur_arm_lists_them_and_keeps_none_of_its_own(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        for mixin in (PlannedForceTorque, PlannedRobotStatus, PlannedFaultAcknowledge, PlannedControllerProvenance):
            with self.subTest(mixin=mixin.__name__):
                self.assertTrue(issubclass(URRobotArm, mixin))
        for name in ("get_tcp_wrench", "get_joint_torques", "get_robot_status", "controller_provenance",
                     "acknowledge_fault"):
            with self.subTest(verb=name):
                self.assertNotIn(name, vars(URRobotArm))


class TheMixinsReadThroughTheTransportTests(unittest.TestCase):
    def test_force_and_torque_are_the_transports_readings(self) -> None:
        transport = fake_transport()
        transport.script("read_tcp_wrench", Wrench(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, frame=Frame.BASE))
        transport.script("read_joint_torques", [0.5] * 6)
        arm = _arm(transport)
        self.assertEqual(3.0, arm.get_tcp_wrench().fz)
        self.assertEqual((0.5,) * 6, arm.get_joint_torques())

    def test_the_status_is_the_transports_and_a_fault_reads_off_it(self) -> None:
        transport = fake_transport()
        transport.script("read_status", _STOPPED)
        status = _arm(transport).get_robot_status()
        self.assertIs(_STOPPED, status)
        self.assertIs(ArmFaultClass.PROTECTIVE_STOP, ArmFault.from_status(status).kind)

    def test_only_a_protective_stop_is_released_from_software(self) -> None:
        transport = fake_transport()
        arm = _arm(transport)
        self.assertTrue(arm.acknowledge_fault(ArmFault(ArmFaultClass.PROTECTIVE_STOP, "latched")))
        self.assertEqual(1, len(transport.calls("release_protective_stop")))
        with self.assertLogs(arm.logger, logging.WARNING) as logged:
            self.assertFalse(arm.acknowledge_fault(ArmFault(ArmFaultClass.EMERGENCY_STOP, "pressed")))
        self.assertIn("not acknowledged from software", logged.output[0])
        self.assertEqual(1, len(transport.calls("release_protective_stop")), "an e-stop sends no release")

    def test_a_release_the_controller_did_not_take_answers_false(self) -> None:
        transport = fake_transport()
        transport.script("release_protective_stop", False)
        self.assertFalse(_arm(transport).acknowledge_fault(ArmFault(ArmFaultClass.PROTECTIVE_STOP, "latched")))

    def test_provenance_never_raises(self) -> None:
        transport = fake_transport()
        transport.script("read_provenance", ControllerProvenance(serial="20235599999", host="127.0.0.1",
                                                                  is_simulator=True),
                         RuntimeError("dashboard closed"))
        arm = _arm(transport)
        self.assertTrue(arm.controller_provenance().is_simulator)
        self.assertEqual(ControllerProvenance(), arm.controller_provenance())


class TheRefusalTextAsksTheTransportTests(unittest.TestCase):
    def test_a_refusal_carries_the_state_the_transport_reports(self) -> None:
        transport = fake_transport()
        transport.script("read_status", _STOPPED)
        text = PlannedArm(**_pieces(transport))._controller_state_text()
        self.assertEqual(" The controller reports robot mode running and safety mode protective_stop, with a "
                         "protective stop active ('PROTECTIVE_STOP').", text)

    def test_a_transport_that_reports_no_status_adds_nothing(self) -> None:
        """A bare PlannedArm needs no status method of its own: the state text asks the transport, and an arm whose
        transport reports none says nothing rather than raising a second fault."""
        arm = PlannedArm(**_pieces(fake_transport(offers=(ArmTransport,))))
        self.assertEqual("", arm._controller_state_text())


if __name__ == "__main__":
    unittest.main()
