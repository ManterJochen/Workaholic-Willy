"""A UR whose RTDE control script stopped is a fault of the cell, not an arm still in motion (P0e E13, owner 2026-09-22).

Measured on URSim (UR5e, pinned image, 2026-09-22): a ``getInverseKinematics`` request for a shoulder-singular pose
returned ``[]`` and stopped the control script. From then on ``isProgramRunning()`` was False and ``isSteady()`` was
False for as long as it was polled, while the largest joint speed read 0.000000 rad/s: the arm stood still, and
``isSteady()`` said nothing about rest because the script that answers it was gone. A protective stop ends the script
the same way (the UR3e pick-loop probe the same day).

The steady gate (``safety.dwell``) read that False as motion, waited its whole timeout and reported "the arm did not
come to rest", before every motion verb; a pick through the service had read it the same way since the gate lived in
the execution policy. Nothing asked whether the script ran, and with the robot RUNNING and safety NORMAL no fault was
found either, so a campaign went on filing a stopped cell as failed grasps. Now the connection asks
``isProgramRunning()`` after an ``isSteady()`` that says False, the gate reports a CONNECTION_ERROR at once, and the
status carries ``program_running``, from which :meth:`ArmFault.from_status` names the fault ``program_stopped``.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from src.robot.core import (
    ArmFault,
    ArmFaultClass,
    JointPositions,
    MotionStatus,
    RobotConnectionError,
    RobotMode,
    RobotMotionRejected,
    RobotStatus,
    SafetyMode,
)
from src.robot.drivers.ur.connection import URConnection
from src.robot.grasping.loop.pick_loop import PickOutcome
from tests.test_pick_loop_controller_state import _orchestrator, _run, _status, _StatusArm
from tests.test_the_arm_holds_its_own_steady_gate import _JOINTS, _arm, _pose
from tests.test_ur_io_force_status import _CTRL_API, _RECV_API, _ur_arm

_SENT = ("moveJ", "moveL", "getInverseKinematics", "getForwardKinematics")


def _connection(*, steady: bool, running: bool) -> tuple[URConnection, MagicMock]:
    """A real ``URConnection`` over spec'd RTDE interfaces, in the state measured on URSim."""
    conn = URConnection(ip="127.0.0.1")
    ctrl = MagicMock(spec=_CTRL_API)
    ctrl.isSteady.return_value = steady
    ctrl.isProgramRunning.return_value = running
    conn._ctrl, conn._recv = ctrl, MagicMock(spec=_RECV_API)
    return conn, ctrl


class TheConnectionTellsAStoppedScriptFromMotionTests(unittest.TestCase):
    def test_a_stopped_script_raises_at_once_instead_of_waiting(self) -> None:
        conn, _ = _connection(steady=False, running=False)
        started = time.monotonic()
        with self.assertRaisesRegex(RobotConnectionError, r"control script is not running .*127\.0\.0\.1"):
            conn.wait_until_steady(0.5, 0.02)
        self.assertLess(time.monotonic() - started, 0.1)

    def test_a_zero_timeout_asks_the_same(self) -> None:
        conn, _ = _connection(steady=False, running=False)
        with self.assertRaises(RobotConnectionError):
            conn.wait_until_steady(0.0, 0.02)

    def test_a_steady_arm_pays_nothing(self) -> None:
        conn, ctrl = _connection(steady=True, running=False)
        self.assertTrue(conn.wait_until_steady(0.5, 0.02))
        ctrl.isProgramRunning.assert_not_called()

    def test_an_arm_in_motion_with_its_script_running_still_times_out(self) -> None:
        conn, _ = _connection(steady=False, running=True)
        self.assertFalse(conn.wait_until_steady(0.05, 0.01))

    def test_a_script_that_stops_during_the_wait_ends_it(self) -> None:
        conn, ctrl = _connection(steady=False, running=True)
        ctrl.isProgramRunning.side_effect = [True, True, False]
        started = time.monotonic()
        with self.assertRaises(RobotConnectionError):
            conn.wait_until_steady(2.0, 0.01)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_the_connection_says_whether_its_script_runs(self) -> None:
        for running in (True, False):
            with self.subTest(running=running):
                conn, _ = _connection(steady=True, running=running)
                self.assertIs(running, conn.is_program_running())


class EveryVerbReportsTheStoppedScriptTests(unittest.TestCase):
    """The URSim probe's step 4, off the box: the arm stands still and its control script is gone.

    The arm is a planned arm over the fake (P0g slice 13), and its steady gate raises exactly what the real
    ``URConnection`` raises for a stopped script, taken from one (``_connection`` above): the connection's half is
    ``TheConnectionTellsAStoppedScriptFromMotionTests``, the arm's is here.
    """

    @staticmethod
    def _what_a_stopped_script_raises() -> BaseException:
        conn, _ = _connection(steady=False, running=False)
        try:
            conn.wait_until_steady(0.5, 0.02)
        except RobotConnectionError as exc:
            return exc
        raise AssertionError("a stopped script did not raise")

    def test_each_verb_says_connection_error_at_once_and_sends_nothing(self) -> None:
        verbs = {
            "move": lambda arm: arm.move(_pose()),
            "move linear": lambda arm: arm.move(_pose(), linear=True),
            "move_to_joints": lambda arm: arm.move_to_joints(JointPositions(_JOINTS)),
            "move_home": lambda arm: arm.move_home(),
            "move_linear": lambda arm: arm.move_linear(_pose()),
        }
        for label, verb in verbs.items():
            with self.subTest(label):
                arm, fake = _arm()
                fake.always("wait_until_steady", self._what_a_stopped_script_raises())
                started = time.monotonic()
                if label == "move_linear":
                    with self.assertRaises(RobotMotionRejected) as caught:
                        verb(arm)
                    result = caught.exception.result
                elif label == "move_home":
                    self.assertIs(False, verb(arm))
                    result = None
                else:
                    result = verb(arm)
                self.assertLess(time.monotonic() - started, 0.5, "the gate waited on a stopped script")
                if result is not None:
                    self.assertIs(MotionStatus.CONNECTION_ERROR, result.status)  # type: ignore[attr-defined]
                    self.assertIn("control script is not running", result.message)  # type: ignore[attr-defined]
                    self.assertNotIn("did not come to rest", result.message)  # type: ignore[attr-defined]
                    self.assertIn("nothing was sent", result.message)  # type: ignore[attr-defined]
                # Asked once, and nothing after it: no path, no Cartesian move, no IK question.
                self.assertEqual(["wait_until_steady"], [call.verb for call in fake.tape])


def _running_normal(*, program_running: bool, protective: bool = False) -> RobotStatus:
    return RobotStatus(
        robot_mode=RobotMode.RUNNING,
        safety_mode=SafetyMode.PROTECTIVE_STOP if protective else SafetyMode.NORMAL,
        protective_stopped=protective, emergency_stopped=False, message="Safetystatus: NORMAL",
        program_running=program_running,
    )


class TheStatusNamesTheFaultTests(unittest.TestCase):
    def test_a_stopped_program_is_not_operational_and_its_own_fault(self) -> None:
        status = _running_normal(program_running=False)
        self.assertTrue(status.controller_ready)
        self.assertFalse(status.is_operational)
        fault = ArmFault.from_status(status)
        assert fault is not None
        self.assertIs(ArmFaultClass.PROGRAM_STOPPED, fault.kind)
        self.assertFalse(fault.acknowledgeable, "a stopped program is started again by a connect, not cleared")
        self.assertIn("connect the arm again", fault.render())
        self.assertIn("program_running=False", fault.render())

    def test_a_running_program_on_a_healthy_controller_is_no_fault(self) -> None:
        self.assertIsNone(ArmFault.from_status(_running_normal(program_running=True)))

    def test_a_protective_stop_outranks_the_program_it_stopped(self) -> None:
        """Clearing the stop comes first, so the stop is the fault named; the stopped program is its evidence, since
        a connect is still owed once the stop is cleared. Where the program runs, the text is the one before E13."""
        fault = ArmFault.from_status(_running_normal(program_running=False, protective=True))
        assert fault is not None
        self.assertIs(ArmFaultClass.PROTECTIVE_STOP, fault.kind)
        self.assertIn("emergency_stop=False, program_running=False)", fault.render())
        stopped = ArmFault.from_status(_running_normal(program_running=True, protective=True))
        assert stopped is not None
        self.assertIn("emergency_stop=False). Nothing is wrong", stopped.render())
        self.assertNotIn("program_running", stopped.render())

    def test_the_status_says_it_is_required(self) -> None:
        """No silent default: a producer that does not say whether its program runs does not build a status."""
        with self.assertRaises(TypeError):
            RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, False, False)  # type: ignore[call-arg]


class TheUrArmReportsItsProgramTests(unittest.TestCase):
    def test_the_status_reads_whether_the_script_runs(self) -> None:
        for running in (True, False):
            with self.subTest(running=running):
                arm, conn = _ur_arm()
                conn.get_robot_mode.return_value = 7
                conn.get_safety_mode.return_value = 1
                conn.is_protective_stopped.return_value = False
                conn.is_emergency_stopped.return_value = False
                conn.dashboard_safety_status.return_value = "Safetystatus: NORMAL"
                conn.is_program_running.return_value = running
                status = arm.get_robot_status()
                self.assertIs(running, status.program_running)
                self.assertIs(running, status.is_operational)

    def test_a_stopped_program_is_not_acknowledged_from_software(self) -> None:
        arm, conn = _ur_arm()
        self.assertFalse(arm.acknowledge_fault(ArmFault(ArmFaultClass.PROGRAM_STOPPED)))
        conn.unlock_protective_stop.assert_not_called()


class TheCampaignStopsOnAStoppedProgramTests(unittest.TestCase):
    def test_a_failed_motion_on_a_stopped_program_ends_the_run_as_a_fault_of_the_cell(self) -> None:
        """Willy ends a campaign on a stopped cell through its report (``controller_stopped``, the owner-cell audit of
        2026-09-23), not by raising: the outcome says the controller cannot move, in the fault's own words."""
        arm = _StatusArm()
        arm._status = _status(mode=RobotMode.RUNNING, safety=SafetyMode.NORMAL, program_running=False,
                              message="Safetystatus: NORMAL")
        outcome, _reasons = _run(arm)
        self.assertIs(PickOutcome.CONTROLLER_NOT_OPERATIONAL, outcome)
        said = _orchestrator(arm)._controller_cannot_move()  # noqa: SLF001 (the diagnosis the pick logs and reports)
        self.assertIn(ArmFaultClass.PROGRAM_STOPPED.value, said)
        self.assertIn("connect the arm again", said)


if __name__ == "__main__":
    unittest.main()
