"""A cuRobo motion with neither a live camera world nor a decline is refused, on every verb of both drivers.

Owner, Step 6 (.commits/robot/54-a-planner-starts-on-a-measured-combination.md): Q1 A, the refusal is ``UNSUPPORTED``
carrying MISSING; Q2 A, all eight UR verbs and all six sim verbs refuse, ``move`` and ``move_to_joints`` keep the
keyword and the rest decline through the block; the refusal is the first statement of each verb, so a refused motion
reads no connection, builds no planner, starts no sidecar and runs no guard. A planner start is not a motion and is not
refused. The mock, ik and RMPflow say UNPLANNED and never refuse for the camera world.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

from src.geometry import Frame, Pose
from src.robot.core import (
    NO_CAMERA_WORLD_MESSAGE,
    CameraWorldUse,
    JointPositions,
    MotionStatus,
    RobotMotionRejected,
)
from src.robot.safety import SafetyPreflight
from tests.test_camera_world_on_every_driver import (
    _JOINTS,
    _AcceptingGuard,
    _pose,
    _sim_mock,
    _sim_unconnected,
    _ur_arm,
    _ur_curobo,
)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, name: str):  # noqa: ANN204
        def record(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            self.calls.append(name)
            raise AssertionError(f"{name} was reached by a motion that should have been refused")
        return record


def _refused_ur() -> "tuple[object, _Recorder, MagicMock]":
    planner = MagicMock()
    arm = _ur_curobo(planner)
    conn = MagicMock()
    conn.is_connected = True
    arm._conn = conn
    recorder = _Recorder()
    arm._home_is_admissible = recorder("_home_is_admissible")  # type: ignore[method-assign]
    arm._home_refusal = recorder("_home_refusal")  # type: ignore[method-assign]
    arm._judge_joint_move = recorder("_judge_joint_move")  # type: ignore[method-assign]
    arm._judge_linear_move = recorder("_judge_linear_move")  # type: ignore[method-assign]
    return arm, recorder, planner


class EveryUrVerbRefusesTests(unittest.TestCase):
    def test_every_ur_verb_refuses_without_a_world_or_a_decline(self) -> None:
        arm, recorder, planner = _refused_ur()

        moved = arm.move(_pose())
        self.assertIs(moved.status, MotionStatus.UNSUPPORTED)
        self.assertIs(moved.camera_world.use, CameraWorldUse.MISSING)
        self.assertEqual(moved.message, NO_CAMERA_WORLD_MESSAGE)

        joints = arm.move_to_joints(JointPositions(_JOINTS.tolist()))
        self.assertIs(joints.status, MotionStatus.UNSUPPORTED)
        self.assertIs(joints.camera_world.use, CameraWorldUse.MISSING)

        self.assertFalse(arm.move_to(_pose()))
        self.assertFalse(arm.move_home())
        home = arm.move_to_home()
        self.assertIs(home.status, MotionStatus.UNSUPPORTED)
        self.assertIs(home.camera_world.use, CameraWorldUse.MISSING)
        self.assertEqual(home.message, NO_CAMERA_WORLD_MESSAGE)
        self.assertFalse(asyncio.run(arm.amove_to(_pose())))
        self.assertFalse(asyncio.run(arm.amove_home()))

        for verb, call in (("move_joint", lambda: arm.move_joint(JointPositions(_JOINTS.tolist()))),
                           ("move_linear", lambda: arm.move_linear(_pose()))):
            with self.subTest(verb=verb), self.assertRaises(RobotMotionRejected) as caught:
                call()
            self.assertIs(caught.exception.result.camera_world.use, CameraWorldUse.MISSING)

        camera_frame = Pose.identity(Frame.CAMERA, label="cam")
        self.assertIs(arm.move(camera_frame).status, MotionStatus.UNSUPPORTED)

        self.assertEqual(recorder.calls, [])
        planner.plan.assert_not_called()
        planner.execute.assert_not_called()
        self.assertEqual([c for c in arm._conn.method_calls if c[0] in ("moveJ", "moveL")], [])

    def test_a_refused_motion_starts_no_planner(self) -> None:
        built: list[int] = []
        arm = _ur_arm("curobo")
        arm._curobo_client_factory = lambda: built.append(1)  # type: ignore[assignment]
        conn = MagicMock()
        conn.is_connected = True
        arm._conn = conn
        self.assertIs(arm.move(_pose()).status, MotionStatus.UNSUPPORTED)
        self.assertEqual(built, [])
        self.assertIsNone(arm._curobo_ur)
        arm.set_wrist_bodies(())

    def test_a_planner_start_is_not_a_motion(self) -> None:
        """The control: starting a planner commands nothing and is not refused for the camera world."""
        arm = _ur_arm("curobo")
        reached: list[str] = []

        class _Glue:
            def start(self):  # noqa: ANN202
                reached.append("start")
                return "identity"

        arm._curobo_ur_planner = lambda: _Glue()  # type: ignore[method-assign]
        self.assertEqual(arm.start_planner(), "identity")
        self.assertEqual(reached, ["start"])

    def test_a_declined_motion_runs_and_says_so_on_every_verb(self) -> None:
        """The control: inside a block every verb runs and the typed results say DECLINED, a worker thread included."""
        arm = _ur_curobo()
        arm._conn = MagicMock()
        arm._conn.is_connected = True
        arm._conn.moveJ.return_value = True
        arm._home_refusal = lambda joints: None  # type: ignore[method-assign]
        arm._judge_joint_move = lambda joints, command: None  # type: ignore[method-assign]
        arm._judge_linear_move = lambda pose, command: None  # type: ignore[method-assign]
        arm._motion.move_to = lambda *a, **k: True  # type: ignore[method-assign]
        with arm.without_camera_world("bench check, no cameras mounted"):
            self.assertIs(arm.move(_pose()).camera_world.use, CameraWorldUse.DECLINED)
            self.assertIs(arm.move_to_joints(JointPositions(_JOINTS.tolist())).camera_world.use,
                          CameraWorldUse.DECLINED)
            self.assertTrue(arm.move_to(_pose()))
            self.assertTrue(arm.move_home())
            self.assertIs(arm.move_to_home().camera_world.use, CameraWorldUse.DECLINED)
            self.assertTrue(asyncio.run(arm.amove_to(_pose())))
            self.assertTrue(asyncio.run(arm.amove_home()))
            arm.move_joint(JointPositions(_JOINTS.tolist()))
            arm.move_linear(_pose())
        self.assertIs(arm.move(_pose()).status, MotionStatus.UNSUPPORTED)


class EverySimVerbRefusesTests(unittest.TestCase):
    def _variants(self):  # noqa: ANN202
        bare = _sim_unconnected()
        guarded = _sim_unconnected()
        guarded._preflight = SafetyPreflight([_AcceptingGuard()])
        return (("no preflight", bare), ("a preflight", guarded))

    def test_the_sim_refuses_without_a_world_or_a_decline(self) -> None:
        for label, arm in self._variants():
            with self.subTest(arm=label):
                drove: list[str] = []
                arm._drive_joints = lambda joints, **_: drove.append("joints")  # type: ignore[method-assign]
                arm._arm_subset = object()  # type: ignore[assignment]
                moved = arm.move(_pose())
                self.assertIs(moved.status, MotionStatus.UNSUPPORTED)
                self.assertIs(moved.camera_world.use, CameraWorldUse.MISSING)
                self.assertIs(arm.move_to_joints(JointPositions(_JOINTS.tolist())).status, MotionStatus.UNSUPPORTED)
                with self.assertRaises(RobotMotionRejected) as caught:
                    arm.move_joint(JointPositions(_JOINTS.tolist()))
                self.assertIn("no live camera world", str(caught.exception))
                self.assertIs(caught.exception.result.status, MotionStatus.UNSUPPORTED)
                with self.assertRaises(RobotMotionRejected) as lined:
                    arm.move_linear(_pose())
                self.assertIs(lined.exception.result.camera_world.use, CameraWorldUse.MISSING)
                self.assertFalse(arm.move_to(_pose()))
                self.assertFalse(arm.move_home())
                self.assertEqual(drove, [])

    def test_the_mock_never_refuses_for_the_camera_world(self) -> None:
        """The control: the mock plans nothing and says UNPLANNED."""
        arm = _sim_mock()
        self.assertTrue(arm.move_to(_pose()))
        self.assertTrue(arm.move_home())
        self.assertFalse(arm.camera_world_required)
        self.assertTrue(_sim_unconnected().camera_world_required)


if __name__ == "__main__":
    unittest.main()
