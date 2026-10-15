"""A fault is acknowledged over a link that watches the controller and never drives it (P0i, owner 2026-09-26).

URSim, 2026-09-26 (``scripts/ursim/probe_acknowledge.py``): the door connected the arm fully, which opens the UR control
interface, and a controller in a protective stop starts no external control program, so every run of the door was
refused at its connect, 1 of 4. The owner chose a watch link the same day: an optional transport Protocol,
``WatchesController``, that reads the controller's state and releases a protective stop and never starts a control
program (the UR: RTDE receive and the dashboard). A planned arm whose vendor lists ``PlannedWatching`` offers it as
``SupportsWatching``; while it watches, every motion verb refuses with nothing sent. The door takes the cell lock and
watches, and on a watch no control program runs by design, so a ready controller with none is no fault it clears.
"""

from __future__ import annotations

import unittest
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

from src.robot.core import (
    JointPositions,
    RobotMode,
    RobotStatus,
    SafetyMode,
    SupportsRobotStatus,
    SupportsWatching,
)
from src.robot.core.errors import RobotConnectionError
from src.robot.core.transport import WatchesController
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.drivers.planned.capabilities import (
    PlannedFaultAcknowledge,
    PlannedRobotStatus,
    PlannedWatching,
)
from src.robot.drivers.ur import connection as conn_mod
from src.robot.drivers.ur.transport import URTransport
from src.robot.execution.abilities import Ability, AbilityStatus, row_of
from src.robot.execution.cell_lock import peek
from src.robot.execution.robot import Robot
from src.robot.faults import AcknowledgementOutcome, FaultAcknowledgement
from tests._fake_transport import fake_transport
from tests.test_a_planned_arm_is_built_by_keyword import _pieces

_STOPPED = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP, protective_stopped=True,
                       emergency_stopped=False, message="PROTECTIVE_STOP", program_running=False)
#: A ready controller seen over a watch: nothing latched, and no control program, since a watch never starts one.
_READY_ON_A_WATCH = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                                emergency_stopped=False, message="NORMAL", program_running=False)


class _Watched(PlannedRobotStatus, PlannedFaultAcknowledge, PlannedWatching, PlannedArm):
    """A vendor whose transport reads the status, releases a stop and watches."""


class _NotWatched(PlannedRobotStatus, PlannedFaultAcknowledge, PlannedArm):
    """A vendor whose transport reads the status and releases a stop, and has no link that only watches."""


def _stopped_cell() -> tuple[Any, Any]:
    """A fake controller in a protective stop that refuses a full connect, as URSim did."""
    fake = fake_transport(connected=False)
    fake.always("connect", RobotConnectionError(
        "the controller reports 'SAFETYSTATUS: PROTECTIVE_STOP'. A stopped controller will not start an external "
        "control program"))
    fake.always("read_status", _STOPPED)
    return fake, _Watched(**_pieces(fake))


def _modules(*, dashboard: bool = True, dashboard_connects: bool = True) -> tuple[Any, Any, Any, Any]:
    ctrl = MagicMock()
    recv_mod = MagicMock()
    recv = recv_mod.RTDEReceiveInterface.return_value
    recv.getRobotMode.return_value = 7
    recv.getSafetyMode.return_value = 3
    recv.isProtectiveStopped.return_value = True
    recv.isEmergencyStopped.return_value = False
    dash_mod = MagicMock() if dashboard else None
    dash = None
    if dash_mod is not None:
        dash = dash_mod.DashboardClient.return_value
        dash.safetystatus.return_value = "Safetystatus: PROTECTIVE_STOP"
        if not dashboard_connects:
            dash.connect.side_effect = RuntimeError("dashboard did not answer")
    return ctrl, recv_mod, dash_mod, dash


def _connection() -> conn_mod.URConnection:
    return conn_mod.URConnection(ip="192.0.2.7", vel=0.2, acc=0.5, frequency=500.0)


class TheUrConnectionWatchesWithoutAControlProgramTests(unittest.TestCase):
    def _patched(self, ctrl: Any, recv_mod: Any, dash_mod: Any) -> Any:
        return patch.multiple(conn_mod, rtde_control=ctrl, rtde_receive=recv_mod, rtde_io=MagicMock(),
                              dashboard_client=dash_mod)

    def test_a_watch_opens_receive_and_dashboard_and_never_the_control_interface(self) -> None:
        ctrl, recv_mod, dash_mod, dash = _modules()
        with self._patched(ctrl, recv_mod, dash_mod):
            link = _connection()
            link.connect_watch()
            self.assertEqual(ctrl.RTDEControlInterface.call_count, 0, "a watch started a control program")
            self.assertTrue(link.is_watching)
            self.assertFalse(link.is_connected, "a watch is no connection a motion may use")
            self.assertEqual(link.get_safety_mode(), 3)
            self.assertTrue(link.is_protective_stopped())
            self.assertTrue(link.unlock_protective_stop())
            dash.unlockProtectiveStop.assert_called_once()
            link.disconnect()
        self.assertFalse(link.is_watching)
        recv_mod.RTDEReceiveInterface.return_value.disconnect.assert_called_once()
        dash.disconnect.assert_called_once()

    def test_a_watch_with_no_dashboard_is_refused_and_leaves_nothing_open(self) -> None:
        """A link that cannot release a stop is no watch the door can use."""
        ctrl, recv_mod, dash_mod, _dash = _modules(dashboard=False)
        with self._patched(ctrl, recv_mod, dash_mod):
            link = _connection()
            with self.assertRaises(ImportError):
                link.connect_watch()
        self.assertFalse(link.is_watching)
        self.assertEqual(recv_mod.RTDEReceiveInterface.call_count, 0)

    def test_a_dashboard_that_does_not_answer_leaves_nothing_open(self) -> None:
        ctrl, recv_mod, dash_mod, _dash = _modules(dashboard_connects=False)
        with self._patched(ctrl, recv_mod, dash_mod):
            link = _connection()
            with self.assertRaises(RobotConnectionError) as refused:
                link.connect_watch()
        self.assertIn("192.0.2.7", str(refused.exception))
        self.assertFalse(link.is_watching)
        recv_mod.RTDEReceiveInterface.return_value.disconnect.assert_called_once()

    def test_the_status_over_a_watch_says_no_control_program_runs(self) -> None:
        ctrl, recv_mod, dash_mod, _dash = _modules()
        with self._patched(ctrl, recv_mod, dash_mod):
            link = _connection()
            link.connect_watch()
            status = URTransport(link, MagicMock()).read_status()
        self.assertIs(status.safety_mode, SafetyMode.PROTECTIVE_STOP)
        self.assertTrue(status.protective_stopped)
        self.assertFalse(status.program_running)
        self.assertEqual(ctrl.RTDEControlInterface.call_count, 0)

    def test_the_ur_transport_offers_the_watch(self) -> None:
        self.assertIsInstance(URTransport(MagicMock(), MagicMock()), WatchesController)


class APlannedArmWatchesTests(unittest.TestCase):
    def test_a_watch_opens_the_watch_link_and_never_the_control_link(self) -> None:
        fake, arm = _stopped_cell()
        arm.watch()
        self.assertEqual(len(fake.calls("connect_watch")), 1)
        self.assertEqual(fake.calls("connect"), [])
        self.assertTrue(arm.is_watching)
        self.assertFalse(arm.is_connected)
        self.assertTrue(arm.get_robot_status().protective_stopped)

    def test_every_motion_verb_refuses_while_watching_and_sends_nothing(self) -> None:
        fake, arm = _stopped_cell()
        arm.watch()
        result = arm.move_to_joints(JointPositions.from_list([0.0, -1.2, 1.3, -0.4, 1.5, 0.2]))
        self.assertFalse(result.ok)
        self.assertEqual(fake.calls("run_trajectory"), [])

    def test_disconnect_ends_the_watch(self) -> None:
        fake, arm = _stopped_cell()
        arm.watch()
        arm.disconnect()
        self.assertFalse(arm.is_watching)
        self.assertEqual(len(fake.calls("disconnect")), 1)

    def test_only_an_arm_whose_vendor_lists_it_can_be_watched(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        self.assertNotIsInstance(PlannedArm(**_pieces(fake_transport())), SupportsWatching)
        self.assertNotIsInstance(_NotWatched(**_pieces(fake_transport())), SupportsWatching)
        self.assertIsInstance(_Watched(**_pieces(fake_transport())), SupportsWatching)
        self.assertTrue(issubclass(URRobotArm, PlannedWatching))


class TheDoorWatchesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = f"test-watch-{uuid.uuid4().hex}"

    def test_a_protective_stop_is_acknowledged_where_a_full_connect_is_refused(self) -> None:
        fake, arm = _stopped_cell()
        robot = Robot.from_parts(arm=arm, gripper=None, lock_key=self.key)
        held: list[bool] = []

        def confirm(_sentence: str) -> bool:
            held.append(peek(self.key) is not None)
            return True

        report = FaultAcknowledgement(robot=robot, confirm=confirm).run()
        self.assertIs(report.outcome, AcknowledgementOutcome.ACKNOWLEDGED, report.render())
        self.assertEqual(held, [True], "the person was asked while the door held the cell lock")
        self.assertEqual(fake.calls("connect"), [], "the door asked for a control program")
        self.assertEqual(len(fake.calls("release_protective_stop")), 1)
        self.assertFalse(arm.is_watching, "the watch is closed on the way out")
        self.assertIsNone(peek(self.key), "the lock is given back on the way out")

    def test_a_ready_controller_is_no_fault_although_no_control_program_runs_on_a_watch(self) -> None:
        fake, arm = _stopped_cell()
        fake.always("read_status", _READY_ON_A_WATCH)
        report = FaultAcknowledgement(robot=Robot.from_parts(arm=arm, gripper=None, lock_key=self.key),
                                      confirm=lambda _sentence: True).run()
        self.assertIs(report.outcome, AcknowledgementOutcome.NO_FAULT, report.render())
        self.assertEqual(fake.calls("release_protective_stop"), [])

    def test_a_watch_that_does_not_open_gives_the_lock_back_and_closes_what_it_opened(self) -> None:
        fake, arm = _stopped_cell()
        fake.always("connect_watch", RobotConnectionError("could not watch the UR controller: dashboard did not answer"))
        report = FaultAcknowledgement(robot=Robot.from_parts(arm=arm, gripper=None, lock_key=self.key),
                                      confirm=lambda _sentence: True).run()
        self.assertIs(report.outcome, AcknowledgementOutcome.REFUSED)
        self.assertIn("dashboard did not answer", report.detail)
        self.assertIsNone(peek(self.key), "the lock was kept after a watch that did not open")
        self.assertEqual(len(fake.calls("disconnect")), 1)
        self.assertEqual(fake.calls("release_protective_stop"), [])

    def test_an_arm_that_cannot_be_watched_is_refused_before_the_lock_is_taken(self) -> None:
        arm = _NotWatched(**_pieces(fake_transport(connected=False)))
        with self.assertRaises(TypeError) as refused:
            with Robot.from_parts(arm=arm, gripper=None, lock_key=self.key).watching():
                self.fail("an arm that cannot be watched was watched")
        self.assertIn("SupportsWatching", str(refused.exception))
        self.assertIsNone(peek(self.key))

    def test_an_arm_that_cannot_be_watched_is_refused_by_its_row_and_nothing_connects(self) -> None:
        fake = fake_transport(connected=False)
        arm = _NotWatched(**_pieces(fake))
        self.assertIsInstance(arm, SupportsRobotStatus)
        row = row_of(Ability.ACKNOWLEDGE_FAULT, arm=arm)
        self.assertIs(row.status, AbilityStatus.REFUSED)
        self.assertIn("SupportsWatching", row.reason)
        report = FaultAcknowledgement(robot=Robot.from_parts(arm=arm, gripper=None, lock_key=self.key),
                                      confirm=lambda _sentence: True).run()
        self.assertIs(report.outcome, AcknowledgementOutcome.REFUSED)
        self.assertEqual(report.detail, row.reason)
        self.assertEqual([call.verb for call in fake.tape if call.verb.startswith("connect")], [])


if __name__ == "__main__":
    unittest.main()
