"""The UR arm's own: the home its constructor states.

Until P0g slice 13 this file also drove ``move()`` end to end, the function that maps every safety verdict to a typed
MotionStatus (S/P5), and until slice 15 it held connect's payload push and gates. Both are decided in ``PlannedArm``
now, so they are held on the kit's planned arm over the fake transport, in
``tests/test_a_planned_arm_types_every_verdict.py`` and ``tests/test_a_planned_arm_connects_through_its_transport.py``.
What stays here needs the UR driver: its constructor, which chooses the home. ``_arm`` stays for the camera-world file's
check of the UR's own stance.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock


from src.config.schema.robot import RobotConfig
from src.robot.core import MotionStatus
from src.robot.drivers.ur.arm import URRobotArm


def _arm(motion_planner: str = "ik", *, connected: bool = True) -> URRobotArm:
    cfg = RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "127.0.0.1"}, "motion": {"planner": motion_planner}, "gripper": {"model": "robotiq_2f85"}})
    arm = URRobotArm(cfg)
    arm._conn = MagicMock()
    arm._conn.is_connected = connected
    return arm


class TheConstructorStatesItsHomeTests(unittest.TestCase):
    """Which home the UR driver builds its arm with. The home move's box and gates are the planned arm's
    (tests/test_a_planned_arm_types_every_verdict.py)."""

    def test_the_cell_can_declare_its_own_home(self) -> None:
        """The real path had NO field for this; the sim has had one all along."""
        measured_ur3e_home = (3.14, -1.0, 1.0, -1.7, -1.57, 0.0)
        cfg = RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "127.0.0.1"},
                                          "motion": {"home_deg": [math.degrees(v) for v in measured_ur3e_home]},
                                          "gripper": {"model": "robotiq_2f85"}})
        for stated, measured in zip(URRobotArm(cfg)._home_joints, measured_ur3e_home, strict=True):
            self.assertAlmostEqual(stated, measured, places=12)

    def test_an_explicit_argument_still_wins_over_config(self) -> None:
        cfg = RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "127.0.0.1"}, "motion": {"home_deg": (60.0,) * 6}, "gripper": {"model": "robotiq_2f85"}})
        arm = URRobotArm(cfg, home_joints=[0.5] * 6)
        self.assertEqual(list(arm._home_joints), [0.5] * 6)


class AHomeOnAnEndedControlProgramTests(unittest.TestCase):
    """The UR's own home FK (``_home_fk``): after a stop on the pendant the control program has ended and the controller
    reads no FK, so a home a person asks for starts a fresh program and reads it again. Moved here from
    tests/test_a_refused_home_names_its_gate.py when that file moved onto the kit (P0g slice 13): the restart is the
    UR's own."""

    _HERE = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)
    #: What the fake controller's FK answers for a home inside the shipped box: (400, 0, 300) mm, tool down.
    _INSIDE = [0.4, 0.0, 0.3, 0.0, 3.14159, 0.0]

    def _arm(self) -> URRobotArm:
        """A cuRobo UR with the real guards on a fake controller that stands at ``_HERE`` and completes every moveJ."""
        from tests._arm_kit import RecordingPlanner

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"ip": "127.0.0.1"},
            "motion": {"planner": "curobo", "home_deg": [math.degrees(v) for v in (0.2, -1.5, 1.5, 0.0, 0.0, 0.0)]},
            "safety": {},
            "gripper": {"model": "robotiq_2f85", "tool_frame": {
                "source": "willy", "offset_mm": (0.0, 132.0, 0.0),
                "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)}},
        })
        arm = URRobotArm(cfg)
        arm._conn = MagicMock()
        arm._conn.is_connected = True
        arm._conn.get_joint_positions.return_value = list(self._HERE)
        arm._conn.moveJ.return_value = True
        arm._curobo = RecordingPlanner()  # type: ignore[assignment]
        return arm

    @staticmethod
    def _home(arm: URRobotArm) -> object:
        from src.robot.execution.robot import Robot

        return Robot.from_parts(arm=arm, gripper=None, lock_key=None).home(decline="bench check, no cameras mounted")

    def test_a_home_after_the_control_program_ended_runs_on_a_fresh_one(self) -> None:
        """The cell, 2026-10-12: after a stop on the pendant every home read no FK and was refused until the cell was
        taken down. A home a person asks for now starts a fresh control program and reads the FK again."""
        arm = self._arm()
        arm._conn.fk.side_effect = [RuntimeError("getForwardKinematics() function did not succeed!"), list(self._INSIDE)]
        arm._conn.is_protective_stopped.return_value = False
        arm._conn.is_emergency_stopped.return_value = False
        arm._conn.is_program_running.return_value = False

        report = self._home(arm)

        self.assertTrue(report.ok, report.render())
        arm._conn.restart_control_script.assert_called_once_with()
        arm._conn.moveJ.assert_called()

    def test_a_home_in_a_protective_stop_says_to_clear_it_and_restarts_nothing(self) -> None:
        arm = self._arm()
        arm._conn.fk.side_effect = RuntimeError("getForwardKinematics() function did not succeed!")
        arm._conn.is_protective_stopped.return_value = True
        arm._conn.is_emergency_stopped.return_value = False
        arm._conn.is_program_running.return_value = False

        report = self._home(arm)

        self.assertIs(MotionStatus.CONTROLLER_REJECTED, report.status, report.render())
        self.assertIn("protective stop: clear it at the pendant", report.message)
        arm._conn.restart_control_script.assert_not_called()
        arm._conn.moveJ.assert_not_called()

    def test_a_program_that_still_runs_is_not_restarted(self) -> None:
        arm = self._arm()
        arm._conn.fk.side_effect = RuntimeError("controller dropped")
        arm._conn.is_protective_stopped.return_value = False
        arm._conn.is_emergency_stopped.return_value = False
        arm._conn.is_program_running.return_value = True

        report = self._home(arm)

        self.assertIs(MotionStatus.CONTROLLER_REJECTED, report.status, report.render())
        arm._conn.restart_control_script.assert_not_called()


if __name__ == "__main__":
    unittest.main()
