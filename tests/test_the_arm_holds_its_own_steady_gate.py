"""The planned arm holds the steady gate itself, once per motion verb (P0e E9, owner 2026-09-22).

Every arm stands still before it moves (``robot_arm.STEADY_TIMEOUT_S``, the owner's cell's ``safety.dwell`` until
2026-10-17). Until P0e three layers above the arm waited for it: the
Robot motion verbs, the pick-and-place steps and the grasp execution policy. A motion none of them wrapped (a park, a
calibration pose, a recovery move) never waited, and a pick waited in the policy while the arm it drove did not know.
The gate is the arm's own now: every motion verb waits once, after the frame and camera-world refusals and before
anything is planned or sent, and a timeout is the arm's TIMEOUT result. Two things read differently, by the owner's
choice: a timeout on ``Robot.move`` is MOTION_REFUSED where it was REFUSED, and every joint move waits on a real UR.

Since P0g slice 13 these tests stand on a planned arm over the fake transport (``tests/_arm_kit.steady_arm``) and read
what it was sent from the fake's tape, so they hold for every vendor's transport and not through one vendor's wire.
Every "nothing was sent" is anchored on something the tape must hold, so a transport that stopped recording fails here.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    CameraWorldDecline,
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotMotionRejected,
)
from src.robot.core.robot_arm import STEADY_TIMEOUT_S
from tests._arm_kit import HERE, steady_arm

_JOINTS = list(HERE)
#: The ik arm's joint lines ask for a camera world or a decline (``CameraWorldDecline``); this test reads the gate.
_DECLINED = CameraWorldDecline("unit double: this test reads the steady gate and no camera world is wired")
#: What reaches a controller: a joint path, a Cartesian move after its own IK, a question to its IK.
_SENDS = ("run_trajectory", "move_pose", "controller_ik")


def _pose(frame: Frame = Frame.BASE) -> Pose:
    return Pose(position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                frame=frame)


def _arm(*, steady: bool = True, connected: bool = True) -> tuple[Any, Any]:
    """The kit's steady arm and its fake transport, whose steady gate answers ``steady``."""
    arm = steady_arm(steady=steady, connected=connected)
    return arm, arm._transport


def _verbs() -> dict[str, object]:
    return {
        "move": lambda arm: arm.move(_pose()),
        "move linear": lambda arm: arm.move(_pose(), linear=True),
        "move_to_joints": lambda arm: arm.move_to_joints(JointPositions(_JOINTS)),
        "move_home": lambda arm: arm.move_home(),
        "move_linear": lambda arm: arm.move_linear(_pose()),
    }


#: Willy's two joint lines. On this ik arm both are refused once judged, nobody judging its paths, so they send nothing.
_JOINT_LINES = {
    "move_to_joints_on_the_line": lambda arm: arm.move_to_joints_on_the_line(
        JointPositions(_JOINTS), camera_world=_DECLINED),
    "move_through_joints_on_the_line": lambda arm: arm.move_through_joints_on_the_line(
        [JointPositions(_JOINTS)], camera_world=_DECLINED),
}


def _sent(arm: Any, conn: Any) -> list[str]:
    """Every command that reached the controller, and the steady gate's question, in order, from the fake's tape."""
    return [call.verb for call in conn.tape if call.verb in (*_SENDS, "wait_until_steady")]


def _asked_nothing_but_a_control_read(conn: Any) -> list[str]:
    """What the tape holds once the test itself reads the joints: exactly that read when the arm asked nothing."""
    conn.read_joints()
    return [call.verb for call in conn.tape]


class EveryVerbWaitsOnceBeforeItSendsTests(unittest.TestCase):
    def test_each_verb_waits_once_and_first(self) -> None:
        for label, verb in _verbs().items():
            with self.subTest(label):
                arm, conn = _arm()
                verb(arm)  # type: ignore[operator]
                self.assertEqual([(STEADY_TIMEOUT_S, 0.02)], [call.args for call in conn.calls("wait_until_steady")])
                sent = _sent(arm, conn)
                self.assertEqual("wait_until_steady", sent[0], sent)
                self.assertGreater(len(sent), 1, "nothing was sent after the wait")

    def test_a_timeout_sends_nothing_and_says_why(self) -> None:
        for label, verb in _verbs().items():
            with self.subTest(label):
                arm, conn = _arm(steady=False)
                if label == "move_linear":
                    with self.assertRaises(RobotMotionRejected) as caught:
                        verb(arm)  # type: ignore[operator]
                    result = caught.exception.result
                    assert result is not None
                    self.assertIs(MotionStatus.TIMEOUT, result.status)
                elif label == "move_home":
                    self.assertIs(False, verb(arm))  # type: ignore[operator]
                else:
                    result = verb(arm)  # type: ignore[operator]
                    self.assertIs(MotionStatus.TIMEOUT, result.status)  # type: ignore[attr-defined]
                    self.assertIn("did not come to rest within 5.00 s", result.message)  # type: ignore[attr-defined]
                self.assertEqual(["wait_until_steady"], _sent(arm, conn))

    def test_willys_joint_lines_wait_once_before_they_are_judged(self) -> None:
        for label, verb in _JOINT_LINES.items():
            with self.subTest(label):
                arm, conn = _arm()
                self.assertIs(MotionStatus.UNSUPPORTED, verb(arm).status)  # judged after the gate, and refused
                self.assertEqual([(STEADY_TIMEOUT_S, 0.02)], [call.args for call in conn.calls("wait_until_steady")])
                arm, conn = _arm(steady=False)
                result = verb(arm)
                self.assertIs(MotionStatus.TIMEOUT, result.status)
                self.assertIn("did not come to rest within 5.00 s", result.message)
                self.assertEqual(["wait_until_steady"], _sent(arm, conn))

    def test_a_pose_outside_base_is_refused_before_the_wait(self) -> None:
        arm, conn = _arm()
        self.assertIs(MotionStatus.INVALID_TARGET, arm.move(_pose(Frame.TOOL)).status)
        self.assertEqual(["read_joints"], _asked_nothing_but_a_control_read(conn))

    def test_a_closed_link_is_not_waited_on(self) -> None:
        arm, conn = _arm(connected=False)
        arm.move(_pose())
        asked = _asked_nothing_but_a_control_read(conn)
        self.assertNotIn("wait_until_steady", asked)
        self.assertEqual("read_joints", asked[-1], "the tape recorded nothing, so what it lacks says nothing")


class AnArmThatGatesItsOwnSendsWaitsAtTheSendTests(unittest.TestCase):
    """Every cuRobo arm (the owner's cell's ``safety.dwell.gate_at: send`` until 2026-10-17) judges first and waits
    right before it sends
    (``tests/test_a_motion_waits_for_steady_right_before_it_is_sent.py``), so the start of its verb does not wait."""

    def test_the_start_of_its_verb_does_not_wait(self) -> None:
        arm, conn = _arm()
        arm._gates_sends = True  # what every cuRobo arm holds
        self.assertIsNone(arm._steady_refusal(MotionCommand.MOVE_TO))
        self.assertEqual(["read_joints"], _asked_nothing_but_a_control_read(conn))


class TheRobotReadsTheArmsTimeoutTests(unittest.TestCase):
    def test_a_timeout_on_robot_move_is_the_arms_motion_refused(self) -> None:
        """The outcome that changed: MOTION_REFUSED with the arm's TIMEOUT, where the Robot's own gate said REFUSED."""
        from unittest.mock import patch

        from src.robot.execution.motion import MotionOutcome
        from src.robot.execution.robot import Robot
        from src.robot.safety.preflight import SafetyPreflight
        from tests._arm_kit import line_arm

        # A cuRobo arm, because Robot.move refuses the ik route before the arm is asked. It waits for steady at its send,
        # which the drive stood in here answers as a gate that timed out answers.
        arm = line_arm()
        arm._drive_curobo = MagicMock(return_value=MotionResult.failed(  # type: ignore[method-assign]
            MotionStatus.TIMEOUT, MotionCommand.MOVE_TO,
            message="the arm did not come to rest within 5.00 s before the motion was sent; nothing was sent"))
        with patch.object(SafetyPreflight, "path_judge_refusal", return_value=None):
            report = Robot.from_parts(arm=arm, gripper=None, lock_key=None).move(
                _pose(), decline="unit double: this test reads the steady gate and no camera world is wired")
        self.assertIs(MotionOutcome.MOTION_REFUSED, report.outcome, report.render())
        self.assertIs(MotionStatus.TIMEOUT, report.result.status)
        arm._drive_curobo.assert_called_once()


if __name__ == "__main__":
    unittest.main()
