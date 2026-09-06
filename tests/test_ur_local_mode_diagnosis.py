"""A controller in LOCAL mode must say so, not hide behind an RTDE timeout.

Measured against URSim 5.26.0 on 2026-08-18. `URConnection.connect()` builds `RTDEControlInterface`
FIRST, and on a controller in Local control mode that call fails with:

    Failed to start RTDE data synchronization, before timeout

which names nothing. The actual situation is that the control script IS uploaded -- it is visible in
PolyScope's own log -- but a controller in Local mode never RUNS an externally-sent program, so the
synchronisation that the running script would establish never begins. Six minutes went into that
here, with the arm being a simulator. In September the same message would appear next to a powered
robot, and the answer ("someone has to flip a selector on the pendant") is not derivable from it.

Two invariants, and the second is the one that keeps this honest:

1. When the dashboard PROVES Local mode, connect() raises a message that names the cause, the fix,
   and what the fix costs (Remote mode locks the pendant for motion -- ISO wants one source of
   control, so jogging and freedrive go away until you switch back).
2. When nobody can be asked, the ORIGINAL transport error is re-raised unchanged. An unanswered
   question is not a "no": blaming a mode nobody verified would send an operator to the pendant to
   fix something that was never wrong.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.robot.core import RobotConnectionError
from src.robot.drivers.ur import connection as conn_mod


def _connection() -> conn_mod.URConnection:
    return conn_mod.URConnection(ip="192.168.0.100", vel=0.2, acc=0.5, frequency=500.0)


class LocalModeDiagnosisTests(unittest.TestCase):
    """What connect() says when the control interface refuses."""

    @staticmethod
    def _dash(*, remote: bool = True, safety: str = "Safetystatus: NORMAL") -> MagicMock:
        d = MagicMock()
        d.isInRemoteControl.return_value = remote
        d.safetystatus.return_value = safety
        return d

    def _connect_failing(self, *, dashboard: MagicMock | None):
        """Drive connect() with a control interface that always fails, and a chosen dashboard."""
        ctrl = MagicMock()
        ctrl.RTDEControlInterface.side_effect = RuntimeError(
            "Failed to start RTDE data synchronization, before timeout"
        )
        dash_mod = None
        if dashboard is not None:
            dash_mod = MagicMock()
            dash_mod.DashboardClient.return_value = dashboard
        with patch.object(conn_mod, "rtde_control", ctrl), \
             patch.object(conn_mod, "rtde_receive", MagicMock()), \
             patch.object(conn_mod, "rtde_io", MagicMock()), \
             patch.object(conn_mod, "dashboard_client", dash_mod):
            c = _connection()
            with self.assertRaises(Exception) as caught:  # noqa: PT027 - we assert on the type below
                c.connect()
            return caught.exception

    def test_local_mode_is_named_with_the_fix_and_its_cost(self) -> None:
        exc = self._connect_failing(dashboard=self._dash(remote=False))
        self.assertIsInstance(exc, RobotConnectionError)
        msg = str(exc)
        self.assertIn("LOCAL control mode", msg)
        self.assertIn("Remote Control", msg)          # where to click
        self.assertIn("pendant is locked for motion", msg)  # what it costs
        # The original transport error survives -- a diagnosis must not swallow the evidence.
        self.assertIn("Failed to start RTDE data synchronization", msg)

    def test_remote_mode_re_raises_the_original_error(self) -> None:
        # The robot IS in remote control, so Local mode is not the story and must not be invented.
        exc = self._connect_failing(dashboard=self._dash(remote=True))
        self.assertNotIsInstance(exc, RobotConnectionError)
        self.assertIn("Failed to start RTDE data synchronization", str(exc))

    def test_an_unanswerable_dashboard_re_raises_the_original(self) -> None:
        # Cannot ask => cannot claim. Sending an operator to fix a mode that was never wrong is worse
        # than an opaque error, because it looks like an answer.
        dash = self._dash()
        dash.isInRemoteControl.side_effect = RuntimeError("dashboard busy")
        exc = self._connect_failing(dashboard=dash)
        self.assertNotIsInstance(exc, RobotConnectionError)

    def test_no_dashboard_module_re_raises_the_original(self) -> None:
        exc = self._connect_failing(dashboard=None)
        self.assertNotIsInstance(exc, RobotConnectionError)

    def test_the_diagnostic_dashboard_is_always_closed(self) -> None:
        # It opens its own connection on an error path; leaking it would hold a dashboard socket on a
        # controller that is already unhappy.
        dash = self._dash(remote=False)
        self._connect_failing(dashboard=dash)
        dash.disconnect.assert_called()


class StoppedControllerDiagnosisTests(unittest.TestCase):
    """A cell restarted into an uncleared protective stop must be told, not left guessing.

    Measured against URSim on 2026-08-18: with the controller protective-stopped, `connect()` failed
    with "Failed to start control script, before timeout" -- which names nothing. That is the state a
    real cell is in after someone clears a jam and restarts the software without releasing the stop.
    """

    def test_a_protective_stop_is_named_before_the_remote_control_check(self) -> None:
        # Order matters: a stopped controller ALSO tends to report odd remote-control state, and
        # sending an operator to the Local/Remote selector when the real problem is an active stop
        # wastes the one minute that matters.
        dash = LocalModeDiagnosisTests._dash(remote=False, safety="Safetystatus: PROTECTIVE_STOP")
        exc = LocalModeDiagnosisTests()._connect_failing(dashboard=dash)
        self.assertIsInstance(exc, RobotConnectionError)
        msg = str(exc)
        self.assertIn("PROTECTIVE_STOP", msg)
        self.assertIn("where the arm is VISIBLE", msg)
        self.assertNotIn("LOCAL control mode", msg)

    def test_an_emergency_stop_is_named_too(self) -> None:
        dash = LocalModeDiagnosisTests._dash(safety="Safetystatus: ROBOT_EMERGENCY_STOP")
        exc = LocalModeDiagnosisTests()._connect_failing(dashboard=dash)
        self.assertIsInstance(exc, RobotConnectionError)
        self.assertIn("EMERGENCY", str(exc))

    def test_a_healthy_controller_is_never_blamed(self) -> None:
        exc = LocalModeDiagnosisTests()._connect_failing(dashboard=LocalModeDiagnosisTests._dash())
        self.assertNotIsInstance(exc, RobotConnectionError)


class RemoteControlQueryTests(unittest.TestCase):
    """The three-valued query the console and the cell preflight can use."""

    def test_it_reports_none_without_a_dashboard(self) -> None:
        c = _connection()
        self.assertIsNone(c.is_in_remote_control())

    def test_it_reports_the_dashboard_answer(self) -> None:
        c = _connection()
        c._dashboard = MagicMock()
        c._dashboard.isInRemoteControl.return_value = True
        self.assertIs(c.is_in_remote_control(), True)
        c._dashboard.isInRemoteControl.return_value = False
        self.assertIs(c.is_in_remote_control(), False)

    def test_a_failing_dashboard_is_None_not_False(self) -> None:
        # "I could not ask" and "the answer is no" are different facts, and only one of them should
        # ever stop a cell from running.
        c = _connection()
        c._dashboard = MagicMock()
        c._dashboard.isInRemoteControl.side_effect = RuntimeError("boom")
        self.assertIsNone(c.is_in_remote_control())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
