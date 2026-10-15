"""A cuRobo motion with neither a live camera world nor a decline is refused, on every verb of both drivers.

Owner, Step 6 (.commits/robot/54-a-planner-starts-on-a-measured-combination.md): Q1 A, the refusal is ``UNSUPPORTED``
carrying MISSING; Q2 A, every UR verb and every sim verb refuses, ``move`` and ``move_to_joints`` keep the keyword and
the rest decline through the block (the raising, bool and async twins left on 2026-09-19); the refusal is the first
statement of each verb, so a refused motion reads no connection, builds no planner, starts no sidecar and runs no guard.
A planner start is not a motion and is not refused. The mock, ik and RMPflow say UNPLANNED and never refuse for the
camera world.
"""

from __future__ import annotations

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
from tests._arm_kit import AcceptingGuard, bare_arm, curobo_arm, kit_pose
from tests.test_camera_world_on_every_driver import (
    _JOINTS,
    _sim_mock,
    _sim_unconnected,
)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, name: str):  # noqa: ANN204
        def record(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            self.calls.append(name)
            raise AssertionError(f"{name} was reached by a motion that should have been refused")
        return record


def _refused() -> "tuple[object, _Recorder, MagicMock]":
    """A planned arm on cuRobo over the fake, whose planner, home gate and judges fail the test if reached."""
    planner = MagicMock()
    arm = curobo_arm(planner)
    recorder = _Recorder()
    arm._home_refusal = recorder("_home_refusal")  # type: ignore[method-assign]
    arm._judge_joint_move = recorder("_judge_joint_move")  # type: ignore[method-assign]
    arm._judge_linear_move = recorder("_judge_linear_move")  # type: ignore[method-assign]
    return arm, recorder, planner


class EveryPlannedVerbRefusesTests(unittest.TestCase):
    def test_every_planned_verb_refuses_without_a_world_or_a_decline(self) -> None:
        arm, recorder, planner = _refused()

        moved = arm.move(kit_pose())
        self.assertIs(moved.status, MotionStatus.UNSUPPORTED)
        self.assertIs(moved.camera_world.use, CameraWorldUse.MISSING)
        self.assertEqual(moved.message, NO_CAMERA_WORLD_MESSAGE)

        joints = arm.move_to_joints(JointPositions(_JOINTS.tolist()))
        self.assertIs(joints.status, MotionStatus.UNSUPPORTED)
        self.assertIs(joints.camera_world.use, CameraWorldUse.MISSING)

        self.assertFalse(arm.move_home())
        home = arm.move_to_home()
        self.assertIs(home.status, MotionStatus.UNSUPPORTED)
        self.assertIs(home.camera_world.use, CameraWorldUse.MISSING)
        self.assertEqual(home.message, NO_CAMERA_WORLD_MESSAGE)

        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(kit_pose())
        self.assertIs(caught.exception.result.camera_world.use, CameraWorldUse.MISSING)

        camera_frame = Pose.identity(Frame.CAMERA, label="cam")
        self.assertIs(arm.move(camera_frame).status, MotionStatus.UNSUPPORTED)

        self.assertEqual(recorder.calls, [])
        planner.plan.assert_not_called()
        planner.execute.assert_not_called()
        # Nothing was read or sent. Anchored on one read this test makes itself, so a transport that stopped recording
        # fails here instead of passing on an empty tape.
        arm.get_joint_positions()
        self.assertEqual(["read_joints"], [call.verb for call in arm._transport.tape])

    def test_a_refused_motion_starts_no_planner(self) -> None:
        built: list[int] = []
        arm = bare_arm("curobo")
        arm._curobo_client_factory = lambda: built.append(1)  # type: ignore[assignment]
        self.assertIs(arm.move(kit_pose()).status, MotionStatus.UNSUPPORTED)
        self.assertEqual(built, [])
        self.assertIsNone(arm._curobo)
        arm.set_wrist_bodies(())

    def test_a_planner_start_is_not_a_motion(self) -> None:
        """The control: starting a planner commands nothing and is not refused for the camera world."""
        arm = bare_arm("curobo")
        reached: list[str] = []

        class _Glue:
            def start(self):  # noqa: ANN202
                reached.append("start")
                return "identity"

        arm._curobo_planner = lambda: _Glue()  # type: ignore[method-assign]
        self.assertEqual(arm.start_planner(), "identity")
        self.assertEqual(reached, ["start"])

    def test_a_declined_motion_runs_and_says_so_on_every_verb(self) -> None:
        """The control: inside a block every verb runs and the typed results say DECLINED, a worker thread included."""
        arm = curobo_arm()
        arm._home_refusal = lambda joints: None  # type: ignore[method-assign]
        arm._judge_joint_move = lambda joints, command: (joints, None, None)  # type: ignore[method-assign]
        arm._judge_linear_move = lambda pose, command: None  # type: ignore[method-assign]
        with arm.without_camera_world("bench check, no cameras mounted"):
            self.assertIs(arm.move(kit_pose()).camera_world.use, CameraWorldUse.DECLINED)
            self.assertIs(arm.move_to_joints(JointPositions(_JOINTS.tolist())).camera_world.use,
                          CameraWorldUse.DECLINED)
            self.assertTrue(arm.move_home())
            self.assertIs(arm.move_to_home().camera_world.use, CameraWorldUse.DECLINED)
            arm.move_linear(kit_pose())
        self.assertIs(arm.move(kit_pose()).status, MotionStatus.UNSUPPORTED)


class EverySimVerbRefusesTests(unittest.TestCase):
    def _variants(self):  # noqa: ANN202
        bare = _sim_unconnected()
        guarded = _sim_unconnected()
        guarded._preflight = SafetyPreflight([AcceptingGuard()])
        return (("no preflight", bare), ("a preflight", guarded))

    def test_the_sim_refuses_without_a_world_or_a_decline(self) -> None:
        for label, arm in self._variants():
            with self.subTest(arm=label):
                drove: list[str] = []
                arm._drive_joints = lambda joints, **_: drove.append("joints")  # type: ignore[method-assign]
                arm._arm_subset = object()  # type: ignore[assignment]
                moved = arm.move(kit_pose())
                self.assertIs(moved.status, MotionStatus.UNSUPPORTED)
                self.assertIs(moved.camera_world.use, CameraWorldUse.MISSING)
                joints = arm.move_to_joints(JointPositions(_JOINTS.tolist()))
                self.assertIs(joints.status, MotionStatus.UNSUPPORTED)
                self.assertIn("no live camera world", joints.message)
                with self.assertRaises(RobotMotionRejected) as lined:
                    arm.move_linear(kit_pose())
                self.assertIs(lined.exception.result.camera_world.use, CameraWorldUse.MISSING)
                self.assertFalse(arm.move_home())
                self.assertEqual(drove, [])

    def test_the_mock_never_refuses_for_the_camera_world(self) -> None:
        """The control: the mock plans nothing and says UNPLANNED."""
        arm = _sim_mock()
        self.assertTrue(arm.move(kit_pose()).ok)
        self.assertTrue(arm.move_home())
        self.assertFalse(arm.camera_world_required)
        self.assertTrue(_sim_unconnected().camera_world_required)


if __name__ == "__main__":
    unittest.main()
