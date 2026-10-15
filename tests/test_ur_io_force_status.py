"""UR optional-capability tests: digital/analog I/O, force/torque, robot/safety status.

Mirrors ``test_safety_payload.py``: construct the real objects, inject ``MagicMock`` RTDE
interfaces (``_io`` / ``_recv`` / ``_ctrl`` / ``_dashboard``), and assert the vendor-neutral
capability surface maps onto the right ``ur_rtde`` calls. No native ``ur_rtde`` needed.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch

from src.config.schema.robot import RobotConfig
from src.geometry import Frame
from src.robot.core import (
    DigitalIOPort,
    ArmFault,
    ArmFaultClass,
    RobotMode,
    RobotStatus,
    SafetyMode,
    SupportsDigitalIO,
    SupportsForceTorque,
    SupportsFreedrive,
    SupportsRobotStatus,
    Wrench,
)
from src.robot.core.errors import RobotConnectionError
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.connection import URConnection

# The exact ur_rtde interface methods the driver actually calls (grepped from drivers/ur/*.py). Using
# these as `spec=` turns a renamed or typo'd RTDE call into an AttributeError at test time instead of a
# silently-passing MagicMock -- the day-one bug this suite otherwise could not catch. Arity checking
# against the REAL ur_rtde (create_autospec on the installed SDK) stays a bench task: the SDK is not a
# test dependency and its Windows build is unreliable. Keep these lists in sync with the driver.
_RECV_API = [
    "disconnect", "getActualQ", "getActualQd", "getActualTCPForce", "getActualTCPPose", "getDigitalInState",
    "getDigitalOutState", "getPayload", "getPayloadCog", "getRobotMode", "getSafetyMode",
    "isEmergencyStopped", "isProtectiveStopped",
]
_CTRL_API = [
    "disconnect", "endTeachMode", "getAsyncOperationProgressEx", "getForwardKinematics", "getInverseKinematics", "getInverseKinematicsHasSolution",
    "getJointTorques", "isProgramRunning", "isSteady", "kickWatchdog", "moveJ", "moveL", "setPayload",
    "setWatchdog", "stopJ", "stopL", "stopScript", "teachMode",
]
_IO_API = [
    "disconnect", "isConnected", "setAnalogOutputCurrent", "setAnalogOutputVoltage", "setConfigurableDigitalOut",
    "setStandardDigitalOut", "setToolDigitalOut",
]
_DASH_API = ["closeSafetyPopup", "connect", "disconnect", "safetystatus", "unlockProtectiveStop"]


# --------------------------------------------------------------------------- value objects
class ValueObjectTests(unittest.TestCase):
    def test_wrench_components_and_magnitude(self) -> None:
        w = Wrench(3.0, 4.0, 0.0, 1.0, 2.0, 3.0)
        self.assertEqual(w.force, (3.0, 4.0, 0.0))
        self.assertEqual(w.torque, (1.0, 2.0, 3.0))
        self.assertAlmostEqual(w.force_magnitude, 5.0)
        self.assertIs(w.frame, Frame.BASE)  # base-frame default

    def test_safety_mode_is_stopped(self) -> None:
        self.assertTrue(SafetyMode.PROTECTIVE_STOP.is_stopped)
        self.assertTrue(SafetyMode.ROBOT_EMERGENCY_STOP.is_stopped)
        self.assertTrue(SafetyMode.FAULT.is_stopped)
        self.assertFalse(SafetyMode.NORMAL.is_stopped)
        self.assertFalse(SafetyMode.REDUCED.is_stopped)

    def test_robot_status_operational_flags(self) -> None:
        ok = RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, False, False, program_running=True)
        self.assertTrue(ok.is_operational)
        self.assertFalse(ok.is_stopped)
        stopped = RobotStatus(RobotMode.RUNNING, SafetyMode.PROTECTIVE_STOP, True, False, "Protective stop", program_running=True)
        self.assertFalse(stopped.is_operational)
        self.assertTrue(stopped.is_stopped)


# --------------------------------------------------------------------------- connection wrappers
class URConnectionCapabilityTests(unittest.TestCase):
    def _conn(self) -> tuple[URConnection, MagicMock, MagicMock, MagicMock, MagicMock]:
        conn = URConnection(ip="127.0.0.1")
        io = MagicMock(spec=_IO_API)
        recv = MagicMock(spec=_RECV_API)
        ctrl = MagicMock(spec=_CTRL_API)
        dash = MagicMock(spec=_DASH_API)
        conn._io, conn._recv, conn._ctrl, conn._dashboard = io, recv, ctrl, dash
        return conn, io, recv, ctrl, dash

    def test_set_digital_out_banks(self) -> None:
        conn, io, *_ = self._conn()
        conn.set_digital_out(2, True, "standard")
        io.setStandardDigitalOut.assert_called_once_with(2, True)
        conn.set_digital_out(1, False, "configurable")
        io.setConfigurableDigitalOut.assert_called_once_with(1, False)
        conn.set_digital_out(0, True, "tool")
        io.setToolDigitalOut.assert_called_once_with(0, True)
        with self.assertRaises(ValueError):
            conn.set_digital_out(0, True, "bogus")

    def test_analog_out_voltage_and_current(self) -> None:
        conn, io, *_ = self._conn()
        conn.set_analog_out(0, 3.3, current=False)
        io.setAnalogOutputVoltage.assert_called_once_with(0, 3.3)
        conn.set_analog_out(1, 0.02, current=True)
        io.setAnalogOutputCurrent.assert_called_once_with(1, 0.02)

    def test_a_write_the_controller_did_not_take_answers_false(self) -> None:
        """P0g io1: ur_rtde's setters return False for a write they could not deliver, and that answer was thrown
        away, so a gripper's belief moved on a write that never happened (the 2026-09-23 toggle-cell finding)."""
        conn, io, *_ = self._conn()
        io.setToolDigitalOut.return_value = False
        self.assertIs(False, conn.set_digital_out(0, True, "tool"))
        io.setStandardDigitalOut.return_value = True
        self.assertIs(True, conn.set_digital_out(2, True, "standard"))
        io.setAnalogOutputVoltage.return_value = False
        self.assertIs(False, conn.set_analog_out(0, 3.3))

    def test_a_write_on_an_io_link_that_is_down_is_refused_before_any_setter(self) -> None:
        conn, io, *_ = self._conn()
        io.isConnected.return_value = False
        with self.assertRaises(RuntimeError) as caught:
            conn.set_digital_out(0, True, "tool")
        io.setToolDigitalOut.assert_not_called()
        self.assertIn("not connected", str(caught.exception))

        conn = URConnection(ip="127.0.0.1")
        conn._recv = MagicMock()  # connected for reads, but no _io
        with self.assertRaises(RuntimeError):
            conn.set_digital_out(0, True)

    def test_digital_in_out_global_id_mapping(self) -> None:
        conn, _io, recv, *_ = self._conn()
        recv.getDigitalInState.return_value = True
        self.assertTrue(conn.get_digital_in(3, "standard"))
        recv.getDigitalInState.assert_called_with(3)          # standard: id == pin
        conn.get_digital_in(3, "configurable")
        recv.getDigitalInState.assert_called_with(11)         # configurable: 8 + pin
        conn.get_digital_in(1, "tool")
        recv.getDigitalInState.assert_called_with(17)         # tool: 16 + pin
        conn.get_digital_out(0, "configurable")
        recv.getDigitalOutState.assert_called_with(8)

    def test_force_and_joint_torques(self) -> None:
        conn, _io, recv, ctrl, _dash = self._conn()
        recv.getActualTCPForce.return_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ctrl.getJointTorques.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertEqual(conn.get_tcp_force(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(conn.get_joint_torques(), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_status_reads_and_dashboard(self) -> None:
        conn, _io, recv, _ctrl, dash = self._conn()
        recv.getRobotMode.return_value = 7
        recv.getSafetyMode.return_value = 3
        recv.isProtectiveStopped.return_value = True
        recv.isEmergencyStopped.return_value = False
        dash.safetystatus.return_value = "Safetystatus: PROTECTIVE_STOP"
        self.assertEqual(conn.get_robot_mode(), 7)
        self.assertEqual(conn.get_safety_mode(), 3)
        self.assertTrue(conn.is_protective_stopped())
        self.assertEqual(conn.dashboard_safety_status(), "Safetystatus: PROTECTIVE_STOP")

    def test_unlock_protective_stop_uses_dashboard(self) -> None:
        conn, _io, _recv, _ctrl, dash = self._conn()
        self.assertTrue(conn.unlock_protective_stop())
        dash.closeSafetyPopup.assert_called_once()
        dash.unlockProtectiveStop.assert_called_once()

    def test_unlock_without_dashboard_returns_false(self) -> None:
        conn = URConnection(ip="127.0.0.1")  # no dashboard
        self.assertFalse(conn.unlock_protective_stop())
        self.assertEqual(conn.dashboard_safety_status(), "")

    def test_hand_guiding_primitives_map_onto_teach_mode_and_the_watchdog(self) -> None:
        conn, _io, recv, ctrl, _dash = self._conn()
        ctrl.teachMode.return_value = True
        ctrl.endTeachMode.return_value = True
        ctrl.setWatchdog.return_value = True
        ctrl.kickWatchdog.return_value = False
        ctrl.isProgramRunning.return_value = True
        recv.getActualQd.return_value = [0.0, 0.1, -0.2, 0.0, 0.0, 0.3]
        self.assertTrue(conn.set_watchdog(5.0))
        ctrl.setWatchdog.assert_called_once_with(5.0)
        self.assertTrue(conn.teach_mode())
        self.assertFalse(conn.kick_watchdog())
        self.assertTrue(conn.end_teach_mode())
        self.assertTrue(conn.is_program_running())
        self.assertEqual(conn.get_joint_speeds(), [0.0, 0.1, -0.2, 0.0, 0.0, 0.3])
        # freedriveMode is not in _CTRL_API on purpose: its script lines are $5.10-gated, a no-op on a CB3.
        # reuploadScript neither: on the CB3 URSim it uploaded programs that never ran (restart_control_script).

    def test_a_fresh_control_program_comes_from_a_new_control_interface(self) -> None:
        conn, _io, recv, old, _dash = self._conn()
        new = MagicMock(spec=_CTRL_API)
        with patch.object(URConnection, "_open_control", return_value=new) as opened:
            conn.restart_control_script()
        self.assertEqual(old.method_calls, [call.stopScript(), call.disconnect()])
        opened.assert_called_once_with()
        self.assertIs(conn._ctrl, new)
        self.assertIs(conn._recv, recv, "the receive side is not touched")
        self.assertTrue(conn.is_connected)

    def test_a_new_control_interface_that_does_not_come_up_closes_every_interface(self) -> None:
        conn, io, recv, old, dash = self._conn()
        old.stopScript.side_effect = RuntimeError("RTDE control script is not running!")
        refused = RuntimeError("Failed to start RTDE")
        with patch.object(URConnection, "_open_control", side_effect=refused), \
                self.assertRaisesRegex(RuntimeError, "Failed to start RTDE"):
            conn.restart_control_script()
        old.disconnect.assert_called_once_with()
        for interface in (recv, io, dash):
            interface.disconnect.assert_called_once_with()
        self.assertFalse(conn.is_connected, "a reconnect starts clean")

    def test_ctrl_c_in_the_middle_of_a_restart_leaves_nothing_open(self) -> None:
        conn, _io, recv, old, _dash = self._conn()
        old.disconnect.side_effect = [KeyboardInterrupt, None]  # the Ctrl-C lands once
        with patch.object(URConnection, "_open_control") as opened, self.assertRaises(KeyboardInterrupt):
            conn.restart_control_script()
        opened.assert_not_called()
        self.assertEqual(old.stopScript.call_count, 2, "stopped, and stopped again by the teardown")
        recv.disconnect.assert_called_once_with()
        self.assertFalse(conn.is_connected)

    def test_the_controller_payload_is_read_in_kilograms_and_millimetres(self) -> None:
        conn, _io, recv, *_ = self._conn()
        recv.getPayload.return_value = 1.9
        recv.getPayloadCog.return_value = [0.001, -0.002, 0.06]
        mass, cog = conn.get_payload()  # type: ignore[misc]
        self.assertEqual(mass, 1.9)
        for got, want in zip(cog, (1.0, -2.0, 60.0)):
            self.assertAlmostEqual(got, want)

    def test_a_payload_this_controller_does_not_publish_reads_none(self) -> None:
        conn, _io, recv, *_ = self._conn()
        recv.getPayload.side_effect = RuntimeError("unable to get state data for specified key: payload")
        self.assertIsNone(conn.get_payload())


# --------------------------------------------------------------------------- arm capabilities
def _ur_arm() -> tuple[URRobotArm, MagicMock]:
    """A URRobotArm whose whole RTDE connection is a MagicMock (delegation is what we assert)."""
    arm = URRobotArm(RobotConfig.model_validate({"vendor": "ur", "gripper": {"model": "robotiq_2f85"}}))
    # spec=URConnection: the arm delegates to its connection, so a call to a URConnection method that
    # does not exist (renamed/removed) now fails here instead of returning a happy MagicMock.
    conn = MagicMock(spec=URConnection)
    arm._conn = conn
    return arm, conn


class URArmCapabilityTests(unittest.TestCase):
    def test_arm_advertises_capability_protocols(self) -> None:
        arm, _ = _ur_arm()
        self.assertIsInstance(arm, SupportsDigitalIO)
        self.assertIsInstance(arm, SupportsForceTorque)
        self.assertIsInstance(arm, SupportsRobotStatus)
        self.assertIsInstance(arm, SupportsFreedrive)
        # A plain object advertises none of them.
        self.assertNotIsInstance(object(), SupportsForceTorque)

    def test_io_delegation(self) -> None:
        arm, conn = _ur_arm()
        arm.set_digital_output(4, True, port=DigitalIOPort.CONFIGURABLE)
        conn.set_digital_out.assert_called_once_with(4, True, "configurable")
        arm.set_analog_output(0, 5.0)
        conn.set_analog_out.assert_called_once_with(0, 5.0, False)
        conn.get_digital_in.return_value = True
        self.assertTrue(arm.get_digital_input(2))
        conn.get_digital_in.assert_called_once_with(2, "standard")

    def test_a_write_the_controller_did_not_take_raises_naming_the_bank_and_the_pin(self) -> None:
        """P0g io1: a refused write raises before a caller moves its belief, which is the path jaw_io and vacuum
        already take fail-closed for a write that raises (tests/test_jaw_io_gripper.py)."""
        arm, conn = _ur_arm()
        conn.set_digital_out.return_value = False
        with self.assertRaises(RobotConnectionError) as caught:
            arm.set_digital_output(0, True, port=DigitalIOPort.TOOL)
        said = str(caught.exception)
        self.assertIn("tool", said)
        self.assertIn("output 0", said)
        conn.set_analog_out.return_value = False
        with self.assertRaises(RobotConnectionError):
            arm.set_analog_output(0, 5.0)
        conn.end_output_pulse.return_value = False
        with self.assertRaises(RobotConnectionError) as caught:
            arm.end_output_pulse(1, port=DigitalIOPort.TOOL)
        self.assertIn("output 1 = low", str(caught.exception))

    def test_tcp_wrench_and_joint_torques(self) -> None:
        arm, conn = _ur_arm()
        conn.get_tcp_force.return_value = [10.0, 0.0, 0.0, 0.0, 0.0, 2.0]
        w = arm.get_tcp_wrench()
        self.assertIsInstance(w, Wrench)
        self.assertAlmostEqual(w.force_magnitude, 10.0)
        self.assertIs(w.frame, Frame.BASE)
        conn.get_joint_torques.return_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        self.assertEqual(arm.get_joint_torques(), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def test_robot_status_maps_ints_to_enums(self) -> None:
        arm, conn = _ur_arm()
        conn.get_robot_mode.return_value = 7
        conn.get_safety_mode.return_value = 1
        conn.is_protective_stopped.return_value = False
        conn.is_emergency_stopped.return_value = False
        conn.dashboard_safety_status.return_value = "Normal"
        s = arm.get_robot_status()
        self.assertEqual(s.robot_mode, RobotMode.RUNNING)
        self.assertEqual(s.safety_mode, SafetyMode.NORMAL)
        self.assertTrue(s.is_operational)
        self.assertEqual(s.message, "Normal")

    def test_unknown_mode_ints_fall_back(self) -> None:
        arm, conn = _ur_arm()
        conn.get_robot_mode.return_value = -1     # NO_CONTROLLER (unmapped)
        conn.get_safety_mode.return_value = 12    # AUTOMATIC_MODE_SAFEGUARD_STOP (unmapped)
        conn.is_protective_stopped.return_value = False
        conn.is_emergency_stopped.return_value = False
        conn.dashboard_safety_status.return_value = ""
        s = arm.get_robot_status()
        self.assertEqual(s.robot_mode, RobotMode.UNKNOWN)
        self.assertEqual(s.safety_mode, SafetyMode.UNKNOWN)

    def test_a_protective_stop_reads_as_the_classified_fault(self) -> None:
        """What the deleted ``raise_if_stopped`` raised, asked the way the stack asks it (P0e)."""
        arm, conn = _ur_arm()
        conn.get_robot_mode.return_value = 7
        conn.get_safety_mode.return_value = 3     # PROTECTIVE_STOP
        conn.is_protective_stopped.return_value = True
        conn.is_emergency_stopped.return_value = False
        conn.dashboard_safety_status.return_value = "Protective stop: joint torque"
        fault = ArmFault.from_status(arm.get_robot_status())
        assert fault is not None
        self.assertIs(ArmFaultClass.PROTECTIVE_STOP, fault.kind)
        self.assertTrue(fault.acknowledgeable)
        msg = fault.render()
        self.assertIn("protective_stop", msg)
        self.assertIn("Protective stop: joint torque", msg)  # ENRICHED with controller text
        self.assertIn("clear the stop where the arm is visible", msg)  # and what to do about it

    def test_an_operational_controller_reads_as_no_fault(self) -> None:
        arm, conn = _ur_arm()
        conn.get_robot_mode.return_value = 7
        conn.get_safety_mode.return_value = 1
        conn.is_protective_stopped.return_value = False
        conn.is_emergency_stopped.return_value = False
        conn.dashboard_safety_status.return_value = ""
        self.assertIsNone(ArmFault.from_status(arm.get_robot_status()))

    def test_a_person_acknowledges_a_protective_stop_and_nothing_else(self) -> None:
        """Recovery is never automatic, and only one class is clearable from software at all."""
        arm, conn = _ur_arm()
        conn.unlock_protective_stop.return_value = True
        stopped = ArmFault(ArmFaultClass.PROTECTIVE_STOP, "joint torque")
        self.assertTrue(arm.acknowledge_fault(stopped))
        conn.unlock_protective_stop.assert_called_once()

        for kind in (ArmFaultClass.EMERGENCY_STOP, ArmFaultClass.CONTROLLER_FAULT,
                     ArmFaultClass.LINK_LOST, ArmFaultClass.NO_REMOTE_CONTROL, ArmFaultClass.PROGRAM_STOPPED):
            with self.subTest(kind=kind):
                conn.unlock_protective_stop.reset_mock()
                self.assertFalse(arm.acknowledge_fault(ArmFault(kind)))
                conn.unlock_protective_stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
