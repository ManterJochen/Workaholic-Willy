"""While a freedrive session is open, every motion verb of the UR arm refuses, and says the arm is hand-guided.

Owner decision 2: hand guiding is a vendor capability (``SupportsFreedrive``), the UR arm offers it and the sim arm does
not. A person has their hands on a freed arm, so from the with-block's start to its end nothing this arm is asked to do
may drive it: the refusal is the first thing every verb meets, on the ik path and on the cuRobo path, before a planner,
a guard or the controller is asked anything. The session itself is pinned in
``tests/test_a_hand_guided_arm_is_held_on_every_way_out.py``; this file is the arm around it.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import (
    ControllerPayload,
    JointPositions,
    MotionCommand,
    MotionStatus,
    RobotConnectionError,
    RobotEmergencyStop,
    RobotMotionRejected,
    SupportsFreedrive,
)
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.sim.config import SimRobotConfig
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.freedrive import HAND_GUIDED_MESSAGE

_JOINTS = JointPositions([0.0, -1.5, 1.5, 0.0, 1.5, 0.0])


def _pose() -> Pose:
    return Pose(position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                frame=Frame.BASE)


class _Tripwire:
    """Every name it hands out fails the test when called: a refused motion reaches none of them."""

    def __init__(self) -> None:
        self.reached: list[str] = []

    def __call__(self, name: str):  # noqa: ANN204
        def reached(*args: object, **kwargs: object) -> None:
            self.reached.append(name)
            raise AssertionError(f"{name} was reached by a motion of a hand-guided arm")
        return reached


def _arm(motion_planner: str = "ik", *, robot_mode: int = 7, safety_mode: int = 1,
         protective: bool = False) -> "tuple[URRobotArm, MagicMock, _Tripwire]":
    """A connected UR arm on a controller that reads ``robot_mode``/``safety_mode``, its drive paths tripwired."""
    cfg = RobotConfig.model_validate({"vendor": "ur", "ur": {"motion_planner": motion_planner},
                                      "gripper": {"model": "robotiq_2f85"}})
    arm = URRobotArm(cfg)
    conn = MagicMock()
    conn.is_connected = True
    conn.get_robot_mode.return_value = robot_mode
    conn.get_safety_mode.return_value = safety_mode
    conn.is_protective_stopped.return_value = protective
    conn.is_emergency_stopped.return_value = False
    conn.dashboard_safety_status.return_value = "Safetystatus: PROTECTIVE_STOP" if protective else ""
    conn.get_joint_speeds.return_value = [0.0] * 6
    conn.set_watchdog.return_value = True
    conn.teach_mode.return_value = True
    conn.end_teach_mode.return_value = True
    conn.is_program_running.return_value = True
    arm._conn = conn
    arm._motion.conn = conn
    trip = _Tripwire()
    for name in ("_drive_pose", "_drive_joints", "_home_refusal", "_judge_joint_move", "_judge_linear_move",
                 "_curobo_ur_planner", "ik"):
        setattr(arm, name, trip(name))
    arm._motion.move_to = trip("MotionController.move_to")  # type: ignore[method-assign]
    arm._motion.amove_to = trip("MotionController.amove_to")  # type: ignore[method-assign]
    return arm, conn, trip


class EveryVerbRefusesTests(unittest.TestCase):
    def _assert_refused(self, arm: URRobotArm) -> None:
        for name, result, command in (
            ("move", arm.move(_pose()), MotionCommand.MOVE_TO),
            ("move linear", arm.move(_pose(), linear=True), MotionCommand.MOVE_TO),
            ("move_to_joints", arm.move_to_joints(_JOINTS), MotionCommand.MOVE_JOINTS),
            ("move_to_home", arm.move_to_home(), MotionCommand.MOVE_HOME),
        ):
            with self.subTest(verb=name):
                self.assertIs(result.status, MotionStatus.CONTROLLER_REJECTED)
                self.assertIs(result.command, command)
                self.assertIn(HAND_GUIDED_MESSAGE, result.message)
                self.assertIn("no command reached the controller", result.message)
        self.assertFalse(arm.move_to(_pose()))
        self.assertFalse(arm.move_home())
        self.assertFalse(asyncio.run(arm.amove_to(_pose())))
        self.assertFalse(asyncio.run(arm.amove_home()))
        for name, call in (("move_joint", lambda: arm.move_joint(_JOINTS)),
                           ("move_linear", lambda: arm.move_linear(_pose()))):
            with self.subTest(verb=name), self.assertRaises(RobotMotionRejected) as caught:
                call()
            self.assertIn(HAND_GUIDED_MESSAGE, str(caught.exception))
            self.assertIs(getattr(caught.exception.result, "status", None), MotionStatus.CONTROLLER_REJECTED)

    def test_every_verb_of_an_ik_arm_refuses_while_the_session_is_open_free_or_held(self) -> None:
        arm, conn, trip = _arm("ik")
        with arm.freedrive() as session:
            self._assert_refused(arm)
            session.free()
            self._assert_refused(arm)
            session.hold()
            self._assert_refused(arm)
        self.assertEqual(trip.reached, [])
        conn.moveJ.assert_not_called()
        conn.moveL.assert_not_called()

    def test_every_verb_of_a_curobo_arm_refuses_before_the_planner_or_the_camera_world(self) -> None:
        arm, conn, trip = _arm("curobo")
        with arm.freedrive() as session:
            session.free()
            self._assert_refused(arm)  # no decline in scope: the hand-guided refusal comes before the camera world's
            with arm.without_camera_world("bench check, no cameras mounted"):
                self._assert_refused(arm)
        self.assertEqual(trip.reached, [])
        self.assertIsNone(arm._curobo_ur, "no planner was built")
        conn.moveJ.assert_not_called()

    def test_leaving_the_session_gives_motion_back(self) -> None:
        arm, conn, _ = _arm("ik")
        arm.ik = lambda pose, **_: _JOINTS  # type: ignore[method-assign]
        arm.get_joint_positions = lambda: _JOINTS  # type: ignore[method-assign]
        arm._preflight = None  # type: ignore[assignment]
        driven: list[Pose] = []
        arm._drive_pose = lambda pose, **_: driven.append(pose) or True  # type: ignore[method-assign]
        with arm.freedrive() as session:
            session.free()
        self.assertTrue(arm.move_to(_pose()))
        self.assertEqual(len(driven), 1)
        conn.restart_control_script.assert_called_once_with()


class OpeningASessionTests(unittest.TestCase):
    def test_the_ur_arm_advertises_hand_guiding_and_the_sim_and_dummy_arms_do_not(self) -> None:
        arm, _, _ = _arm()
        self.assertIsInstance(arm, SupportsFreedrive)
        self.assertNotIsInstance(IsaacRobotArm(SimRobotConfig(enabled=True, mock_mode=True)), SupportsFreedrive)
        self.assertNotIsInstance(DummyRobotArm(), SupportsFreedrive)

    def test_nothing_is_freed_when_the_session_is_handed_out(self) -> None:
        arm, conn, _ = _arm()
        session = arm.freedrive()
        self.assertFalse(session.is_free)
        conn.teach_mode.assert_not_called()
        conn.set_watchdog.assert_not_called()
        # Open only inside its with-block, which is what holds the arm again at its end.
        self.assertIsNone(arm._hand_guided_refusal(MotionCommand.MOVE_TO))

    def test_an_arm_that_is_not_connected_is_refused(self) -> None:
        arm, conn, _ = _arm()
        conn.is_connected = False
        with self.assertRaisesRegex(RobotConnectionError, "open connection"):
            arm.freedrive()

    def test_a_stopped_controller_is_refused_as_a_stop(self) -> None:
        arm, _, _ = _arm(safety_mode=3, protective=True)
        with self.assertRaisesRegex(RobotEmergencyStop, "protective_stop"):
            arm.freedrive()

    def test_an_arm_with_its_brakes_engaged_is_refused(self) -> None:
        arm, _, _ = _arm(robot_mode=5)
        with self.assertRaisesRegex(RobotMotionRejected, "robot mode idle"):
            arm.freedrive()

    def test_a_second_session_is_refused_while_one_is_open(self) -> None:
        arm, _, _ = _arm()
        waiting = arm.freedrive()
        with arm.freedrive():
            with self.assertRaisesRegex(RobotMotionRejected, HAND_GUIDED_MESSAGE):
                arm.freedrive()
            with self.assertRaisesRegex(RobotMotionRejected, HAND_GUIDED_MESSAGE):
                waiting.__enter__()
        self.assertIsNone(arm._freedrive)
        with arm.freedrive():  # and one after the other is fine
            pass


class ControllerPayloadTests(unittest.TestCase):
    def test_the_controller_payload_is_read_in_millimetres(self) -> None:
        arm, conn, _ = _arm()
        conn.get_payload.return_value = (1.9, (0.0, 0.0, 60.0))
        self.assertEqual(arm.controller_payload(), ControllerPayload(mass_kg=1.9, cog_mm=(0.0, 0.0, 60.0)))

    def test_a_controller_that_does_not_report_it_reads_none(self) -> None:
        arm, conn, _ = _arm()
        conn.get_payload.return_value = None
        self.assertIsNone(arm.controller_payload())


class TheSampleIsTheArmsTcpTests(unittest.TestCase):
    def test_a_sample_reads_the_tcp_with_the_declared_tool_frame_composed(self) -> None:
        """On a ``willy`` cell the controller runs a bare flange; the sample is the grasp centre, as ``get_tcp_pose``."""
        cfg = RobotConfig.model_validate({
            "vendor": "ur", "gripper": {"model": "robotiq_2f85", "tool_frame": {
                "source": "willy", "offset_mm": [0.0, 0.0, 150.0], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        })
        arm = URRobotArm(cfg)
        conn = MagicMock()
        conn.is_connected = True
        conn.get_robot_mode.return_value = 7
        conn.get_safety_mode.return_value = 1
        conn.is_protective_stopped.return_value = False
        conn.is_emergency_stopped.return_value = False
        conn.dashboard_safety_status.return_value = ""
        conn.is_program_running.return_value = True
        conn.get_joint_positions.return_value = [0.0, -1.5, 1.5, 0.0, 1.5, 0.0]
        conn.get_joint_speeds.return_value = [0.0] * 6
        # The flange 0.6 m up, pointing straight down (180 degrees about base X).
        conn.get_tcp_pose.return_value = [0.4, -0.1, 0.6, np.pi, 0.0, 0.0]
        arm._conn = conn
        arm._motion.conn = conn
        with arm.freedrive() as session:
            sample = session.sample()
        np.testing.assert_allclose(sample.tcp_xyz_mm, (400.0, -100.0, 450.0), atol=1e-9)
        np.testing.assert_allclose(sample.tcp_xyz_mm, arm.get_tcp_pose().position_mm, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
