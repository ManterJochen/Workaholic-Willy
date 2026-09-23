"""A CB3 that refuses the control interface is not told to find a Remote Control menu it does not have.

``URConnection._diagnose_connect_failure`` asked the dashboard ``isInRemoteControl`` with no version check. The Remote and
Local control modes, and the dashboard question about them, exist from PolyScope 5.6 on. Below that ur_rtde 1.6.5 prints
"Warning! isInRemoteControl() function is not supported on the dashboard server for PolyScope versions less than 5.6.0"
(read from the installed library's strings) and hands back an answer anyway, so on the lab's UR10 CB3 (PolyScope 3.15.4)
every failed connect was diagnosed as LOCAL control mode, with directions to a pendant menu PolyScope 3.x does not have.

The question is now asked only of a controller that says it runs 5.6 or later; below that, and where the version cannot
be read, the original transport error is re-raised. The robot mode is read instead where it is one that runs no
external program (powered off, powering on, brakes engaged), on every version. tests/test_ur_local_mode_diagnosis.py
holds the e-Series half, measured on URSim 5.26.0.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.robot.core import RobotConnectionError
from src.robot.drivers.ur import connection as conn_mod

_CB3 = "URSoftware 3.15.4.106291 (Mar 17 2021)"
_E_SERIES = "URSoftware 5.11.1.108318 (Nov 07 2021)"
_BEFORE_REMOTE = "URSoftware 5.5.1.85187 (Nov 18 2019)"
_TRANSPORT = "Failed to start RTDE data synchronization, before timeout"


def _dash(*, version: object = _CB3, remote: bool = False, mode: str = "Robotmode: RUNNING",
          safety: str = "Safetystatus: NORMAL") -> MagicMock:
    """A dashboard whose ``isInRemoteControl`` says False, as ur_rtde answers it below PolyScope 5.6."""
    d = MagicMock()
    d.polyscopeVersion.return_value = version
    d.isInRemoteControl.return_value = remote
    d.robotmode.return_value = mode
    d.safetystatus.return_value = safety
    return d


def _connect_failing(dashboard: MagicMock) -> Exception:
    ctrl = MagicMock()
    ctrl.RTDEControlInterface.side_effect = RuntimeError(_TRANSPORT)
    dash_mod = MagicMock()
    dash_mod.DashboardClient.return_value = dashboard
    with patch.object(conn_mod, "rtde_control", ctrl), \
         patch.object(conn_mod, "rtde_receive", MagicMock()), \
         patch.object(conn_mod, "rtde_io", MagicMock()), \
         patch.object(conn_mod, "dashboard_client", dash_mod):
        c = conn_mod.URConnection(ip="192.168.0.100", vel=0.2, acc=0.5, frequency=125.0)
        try:
            c.connect()
        except Exception as exc:  # noqa: BLE001 (the type is what the tests assert)
            return exc
    raise AssertionError("connect() succeeded with a control interface that always fails")


class ACb3IsNeverToldItIsInLocalModeTests(unittest.TestCase):

    def test_a_cb3_re_raises_the_original_error_and_is_not_asked_the_question(self) -> None:
        dash = _dash()

        exc = _connect_failing(dash)

        self.assertNotIsInstance(exc, RobotConnectionError)
        self.assertIn(_TRANSPORT, str(exc))
        self.assertNotIn("LOCAL control mode", str(exc))
        dash.isInRemoteControl.assert_not_called()

    def test_an_e_series_before_5_6_is_not_asked_either(self) -> None:
        dash = _dash(version=_BEFORE_REMOTE)

        exc = _connect_failing(dash)

        self.assertNotIsInstance(exc, RobotConnectionError)
        dash.isInRemoteControl.assert_not_called()

    def test_a_version_that_cannot_be_read_is_no_evidence(self) -> None:
        for version in ("could not understand: 'PolyscopeVersion'", None, RuntimeError("dashboard busy")):
            with self.subTest(version=repr(version)):
                dash = _dash(version=version)
                if isinstance(version, Exception):
                    dash.polyscopeVersion.side_effect = version

                exc = _connect_failing(dash)

                self.assertNotIsInstance(exc, RobotConnectionError)
                dash.isInRemoteControl.assert_not_called()

    def test_the_control_an_e_series_in_local_mode_is_still_named(self) -> None:
        exc = _connect_failing(_dash(version=_E_SERIES, remote=False))

        self.assertIsInstance(exc, RobotConnectionError)
        self.assertIn("LOCAL control mode", str(exc))


class ARobotModeThatRunsNoProgramIsNamedTests(unittest.TestCase):

    def test_a_powered_off_cb3_is_told_to_power_on_and_release_the_brakes(self) -> None:
        exc = _connect_failing(_dash(mode="Robotmode: POWER_OFF"))

        self.assertIsInstance(exc, RobotConnectionError)
        msg = str(exc)
        self.assertIn("'Robotmode: POWER_OFF'", msg)
        self.assertIn("the arm is powered off", msg)
        self.assertIn("power it on and release the brakes", msg)
        self.assertIn(_TRANSPORT, msg, "the diagnosis must not swallow the evidence")
        self.assertNotIn("LOCAL control mode", msg)

    def test_an_idle_arm_is_told_its_brakes_are_engaged(self) -> None:
        exc = _connect_failing(_dash(mode="Robotmode: IDLE"))

        self.assertIsInstance(exc, RobotConnectionError)
        self.assertIn("its brakes are still engaged", str(exc))

    def test_the_mode_is_named_before_the_remote_question_on_an_e_series(self) -> None:
        dash = _dash(version=_E_SERIES, remote=False, mode="Robotmode: POWER_OFF")

        exc = _connect_failing(dash)

        self.assertIn("the arm is powered off", str(exc))
        self.assertNotIn("LOCAL control mode", str(exc))

    def test_a_running_arm_or_an_unreadable_mode_names_nothing(self) -> None:
        for mode in ("Robotmode: RUNNING", "could not understand: 'robotmode'", RuntimeError("dashboard busy")):
            with self.subTest(mode=repr(mode)):
                dash = _dash(mode=mode if isinstance(mode, str) else "")
                if isinstance(mode, Exception):
                    dash.robotmode.side_effect = mode

                exc = _connect_failing(dash)

                self.assertNotIsInstance(exc, RobotConnectionError)

    def test_a_protective_stop_is_still_named_first(self) -> None:
        exc = _connect_failing(_dash(mode="Robotmode: IDLE", safety="Safetystatus: PROTECTIVE_STOP"))

        self.assertIsInstance(exc, RobotConnectionError)
        self.assertIn("PROTECTIVE_STOP", str(exc))
        self.assertNotIn("brakes are still engaged", str(exc))


class TheRemoteControlQueryOnACb3Tests(unittest.TestCase):

    def _connection(self, dash: MagicMock) -> conn_mod.URConnection:
        c = conn_mod.URConnection(ip="192.168.0.100")
        c._dashboard = dash
        return c

    def test_a_cb3_answers_none_and_is_not_asked(self) -> None:
        dash = _dash()

        self.assertIsNone(self._connection(dash).is_in_remote_control())
        dash.isInRemoteControl.assert_not_called()

    def test_an_e_series_answers_what_its_dashboard_says(self) -> None:
        self.assertIs(False, self._connection(_dash(version=_E_SERIES, remote=False)).is_in_remote_control())
        self.assertIs(True, self._connection(_dash(version=_E_SERIES, remote=True)).is_in_remote_control())

    def test_the_throwaway_probe_proves_nothing_on_a_cb3(self) -> None:
        dash_mod = MagicMock()
        dash_mod.DashboardClient.return_value = _dash()
        with patch.object(conn_mod, "dashboard_client", dash_mod):
            self.assertFalse(conn_mod.URConnection(ip="192.168.0.100")._remote_control_is_off())

    def test_the_version_is_read_off_the_dashboard_text(self) -> None:
        self.assertEqual((3, 15), conn_mod._polyscope_version(_dash(version=_CB3)))
        self.assertEqual((5, 11), conn_mod._polyscope_version(_dash(version=_E_SERIES)))
        self.assertIsNone(conn_mod._polyscope_version(_dash(version="no version here")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
