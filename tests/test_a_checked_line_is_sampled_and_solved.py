"""On a cuRobo arm, a straight line is checked where it is straight, one sample at a time.

`moveL` is the one motion this stack does not plan. The controller runs it, and what it runs is a
straight line of the frame IT holds: in `willy` mode a bare flange, in `controller` mode the declared
tool. Nothing in this repository has ever looked at the configurations that line passes through, and a
line whose two ends are clear can still sweep the elbow through a bin wall.

So on a cuRobo cell the line is sampled where the controller draws it, every sample is solved by the
controller's own inverse kinematics seeded with the previous solution (the same kinematics `moveL`
uses, so the configurations judged are the ones executed), and those configurations go to the guards
and to the planner before one `moveL` is sent.

Two refusals belong to the line and to nothing else. A jump between neighbouring solutions means the
controller changed branch mid line and the arm will take a route nobody sampled. And in `willy` mode an
orientation change means the grasp centre does NOT travel straight even though the flange does, which
is the owner's answer Q8: refuse above half a degree and say that controller mode gives a straight tool
line.

The arm is the kit's (``tests/_arm_kit.line_arm``), a ``PlannedArm`` over the fake transport since P0g slice 13, so
what was sent is read from the fake's tape: a line is one ``move_pose`` with ``linear=True``, a joint move a
``run_trajectory``. Every "nothing was sent" first checks that the tape holds what the line read, so a fake that
recorded nothing fails it (``tests/test_the_kit_tests_read_what_the_fake_records.py``).
"""

from __future__ import annotations

import unittest
from typing import Any, cast

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.core.errors import RobotMotionRejected
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety.planning import JointCheckVerdict
from tests._arm_kit import RecordingIk, RecordingPlanner, line_arm
from tests._fake_transport import Rejects

#: Said where each arm is built: these doubles plan or check with cuRobo and carry no camera world, so every
#: motion they command declines it, as a cuRobo cell must since the world became mandatory.
_DECLINED = "unit double: this test exercises the planner and the guard on a cuRobo arm, and no camera world is wired to it"

_HERE_JOINTS = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)
#: A clear configuration the fake controller hands back for every sample, so the guards accept and the
#: tests are about the sampling rather than about the guards.
_SOLUTION = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)


def _mesh_backend_available() -> bool:
    return mesh_backend_status("ur5e") == "ok"


def _pose(x: float = 400.0, y: float = 0.0, z: float = 300.0, quat=None) -> Pose:
    return Pose(
        position_mm=np.array([x, y, z], dtype=np.float64),
        quaternion_xyzw=(np.array([0.0, 1.0, 0.0, 0.0]) if quat is None else np.asarray(quat)),
        frame=Frame.BASE,
    )


#: The recording planner and IK, held by the kit since P0g slice 13.
_RecordingPlanner = RecordingPlanner
_RecordingIk = RecordingIk

#: The arm every test here stands on, built by the kit since P0g slice 13.
_arm = line_arm


def _ik_calls(arm: PlannedArm) -> list[tuple[Pose, JointPositions | None]]:
    return cast(_RecordingIk, arm.ik).calls


def _fake(arm: PlannedArm) -> Any:
    return arm._transport


def _lines(arm: PlannedArm) -> list[Any]:
    """Every ``move_pose`` the arm sent, each one ``moveL`` when its ``linear`` is true."""
    return _fake(arm).calls("move_pose")


def _assert_one_line(test: unittest.TestCase, arm: PlannedArm, msg: str | None = None) -> None:
    lines = _lines(arm)
    test.assertEqual(1, len(lines), msg)
    test.assertIs(True, lines[0].kwargs["linear"], "the line was sent as a joint-space move")


def _assert_nothing_sent(test: unittest.TestCase, arm: PlannedArm) -> None:
    """No line and no joint move, on a tape that holds what the refused line read (the anchor)."""
    fake = _fake(arm)
    test.assertTrue(fake.tape, "the fake recorded nothing, so 'nothing was sent' would read nothing")
    test.assertEqual([], fake.calls("move_pose"))
    test.assertEqual([], fake.calls("run_trajectory"))


class TheLineIsSampledAndSolvedTests(unittest.TestCase):
    """Every sample is one seeded IK call, and the solutions are what the guards see."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_one_seeded_ik_call_per_sample_in_order(self) -> None:
        arm = _arm()
        self.enterContext(arm.without_camera_world(_DECLINED))
        arm.move_linear(_pose(x=400.0))
        calls = _ik_calls(arm)
        self.assertGreater(len(calls), 3, "the line was solved at its ends and nowhere between")
        # Step 8g: the first sample is solved from the joints the arm stands at, as the sim's has been since 8p.
        first_seed = calls[0][1]
        assert first_seed is not None, "the first sample was solved from nowhere"
        np.testing.assert_allclose(first_seed.tolist(), _HERE_JOINTS, atol=1e-12)
        for index in range(1, len(calls)):
            with self.subTest(sample=index):
                seed = calls[index][1]
                self.assertIsNotNone(seed, "a sample was solved without the previous solution")
                assert seed is not None
                np.testing.assert_allclose(seed.tolist(), _SOLUTION, atol=1e-9)

    def test_the_planner_sees_the_solved_configurations(self) -> None:
        planner = _RecordingPlanner()
        arm = _arm(planner=planner)
        self.enterContext(arm.without_camera_world(_DECLINED))
        arm.move_linear(_pose())
        assert planner.checked is not None
        self.assertEqual(len(_ik_calls(arm)), len(planner.checked))

    def test_exactly_one_movel_and_no_movej(self) -> None:
        arm = _arm()
        self.enterContext(arm.without_camera_world(_DECLINED))
        arm.move_linear(_pose())
        _assert_one_line(self, arm)
        self.assertEqual([], _fake(arm).calls("run_trajectory"))

    def test_a_branch_jump_between_neighbours_refuses_before_anything_moves(self) -> None:
        """The controller changing IK branch mid line is a route nobody sampled."""
        jump = list(_SOLUTION)
        jump[0] += 2.5
        arm = _arm(ik_solutions=[_SOLUTION, _SOLUTION, tuple(jump), _SOLUTION])
        self.enterContext(arm.without_camera_world(_DECLINED))
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose())
        self.assertIn("branch", str(caught.exception).lower())
        _assert_nothing_sent(self, arm)

    def test_a_planner_refusal_stops_the_line(self) -> None:
        planner = _RecordingPlanner(
            JointCheckVerdict(valid=False, first_invalid=2, checked=9, reason="sample 2 of 9")
        )
        arm = _arm(planner=planner)
        self.enterContext(arm.without_camera_world(_DECLINED))
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose())
        self.assertIn("sample 2 of 9", str(caught.exception))
        _assert_nothing_sent(self, arm)


#: One step of the line URSim's own IK solved on 2026-09-18: the elbow and wrist 1 turning against each other while the
#: tool rises 10 mm. The sum |dq| * r of that step is 27 to 31 mm, and the largest move of any link origin is 10.00 mm.
_AGAINST_EACH_OTHER = (0.0, 0.0009, -0.0261, 0.0252, 0.0, 0.0)


class AContinuousLineIsNotABranchTests(unittest.TestCase):
    """Step 8g (owner, 2026-09-18). The line gate refused every step whose sum |dq_i| * r_i exceeded the step bound and
    called it a branch change. The sum bounds how far any point can move, and it adds two joints turning against each
    other as if both carried the arm the same way: on URSim it refused a continuous 50 mm lift, which on a real UR is
    nearly every line, the pick's descent included. A branch is a joint jumping; a step the sum cannot vouch for is
    filled in joint space until it can, so the path gate still judges configurations no further apart than its bound.
    """

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_a_line_whose_joints_work_against_each_other_moves_and_every_judged_step_keeps_the_bound(self) -> None:
        solutions = [tuple(a + k * d for a, d in zip(_SOLUTION, _AGAINST_EACH_OTHER)) for k in range(40)]
        planner = _RecordingPlanner()
        arm = _arm(planner=planner, ik_solutions=solutions)
        self.enterContext(arm.without_camera_world(_DECLINED))

        arm.move_linear(_pose(x=350.0))

        _assert_one_line(self, arm, "a continuous line was refused")
        assert planner.checked is not None
        solved = len(_ik_calls(arm))
        self.assertGreater(len(planner.checked), solved, "no step was filled, so the bound below is not met")
        radii = arm.safety_preflight.joint_radii_mm(arm)  # type: ignore[union-attr]
        bound = arm.safety_preflight.path_step_mm  # type: ignore[union-attr]
        for index, (a, b) in enumerate(zip(planner.checked, planner.checked[1:])):
            with self.subTest(step=index):
                swept = sum(abs(y - x) * r for x, y, r in zip(a, b, radii))
                self.assertLessEqual(swept, bound + 1e-6)

    def test_a_branch_refusal_names_the_joint_that_jumped(self) -> None:
        jump = list(_SOLUTION)
        jump[0] += 2.5
        arm = _arm(ik_solutions=[_SOLUTION, _SOLUTION, tuple(jump), _SOLUTION])
        self.enterContext(arm.without_camera_world(_DECLINED))
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose())
        message = str(caught.exception)
        self.assertIn("joint 1", message)
        self.assertIn("2.50 rad", message)
        _assert_nothing_sent(self, arm)

    def test_a_start_on_another_branch_than_the_arm_is_refused(self) -> None:
        """moveL starts at the joints the arm stands at; a line whose first solution is elsewhere is not that line."""
        other = list(_SOLUTION)
        other[3] += 2.0
        arm = _arm(ik_solutions=[tuple(other), tuple(other)])
        self.enterContext(arm.without_camera_world(_DECLINED))
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose())
        self.assertIn("the joints the arm stands at", str(caught.exception))
        self.assertIn("joint 4", str(caught.exception))
        _assert_nothing_sent(self, arm)

    def test_an_elbow_flip_with_no_jump_is_refused_by_its_chord(self) -> None:
        """The reviewer's case, 2026-09-18: near a stretched elbow the two branches reach one flange pose 0.30 rad apart.

        No joint turns more than the branch bound, and the joint line between them puts the flange 2.29 mm off the line
        moveL runs, where a continuous 10 mm step puts it 0.03 mm off.
        """
        stretched = (0.3, -0.9, 0.15, -1.2, -1.5708, 0.4)
        flipped = tuple(a + d for a, d in zip(stretched, (0.0, 0.144, -0.300, 0.156, 0.0, 0.0)))
        arm = _arm(ik_solutions=[stretched, flipped])
        _fake(arm).joints = list(stretched)
        self.enterContext(arm.without_camera_world(_DECLINED))
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose())
        self.assertIn("off the line moveL runs", str(caught.exception))
        _assert_nothing_sent(self, arm)

    def test_a_line_past_the_sample_cap_is_refused_rather_than_raised(self) -> None:
        from unittest.mock import patch

        solutions = [tuple(a + k * d for a, d in zip(_SOLUTION, _AGAINST_EACH_OTHER)) for k in range(40)]
        arm = _arm(ik_solutions=solutions)
        self.enterContext(arm.without_camera_world(_DECLINED))
        with patch("src.robot.drivers.planned.arm.MAX_PATH_SAMPLES", 3), self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose(x=350.0))
        self.assertIn("more than 3 configurations", str(caught.exception))
        _assert_nothing_sent(self, arm)

    def test_a_step_just_under_the_branch_bound_is_filled_and_moves(self) -> None:
        """The edge of the rule: 0.3 rad on wrist 3 in one step is not a branch, and it is filled, not refused."""
        wrist = list(_SOLUTION)
        wrist[5] += 0.3
        planner = _RecordingPlanner()
        arm = _arm(planner=planner, ik_solutions=[_SOLUTION, tuple(wrist)])
        self.enterContext(arm.without_camera_world(_DECLINED))
        arm.move_linear(_pose(x=305.0))
        _assert_one_line(self, arm)
        assert planner.checked is not None
        self.assertGreater(len(planner.checked), len(_ik_calls(arm)))


class TheLineIsSampledWhereTheControllerDrawsItTests(unittest.TestCase):
    """`willy` runs a bare flange, `controller` runs the declared tool, and they are different lines."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_controller_samples_the_tool_line(self) -> None:
        arm = _arm(tool_source="controller")
        self.enterContext(arm.without_camera_world(_DECLINED))
        arm.move_linear(_pose(x=500.0))
        poses = [call[0] for call in _ik_calls(arm)]
        xs = [float(p.position_mm[0]) for p in poses]
        self.assertAlmostEqual(500.0, xs[-1], places=6, msg="the line does not end at the goal")
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(xs, xs[1:])), "the line doubles back")

    def test_willy_refuses_a_turn_because_the_grasp_centre_would_not_travel_straight(self) -> None:
        """The owner's answer Q8. A straight flange line with a turn is a curved tool line."""
        turned = np.array([0.0, 0.9659258, 0.0, 0.2588190])  # 30 degrees about Y from the start
        arm = _arm(tool_source="willy")
        self.enterContext(arm.without_camera_world(_DECLINED))
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose(quat=turned))
        message = str(caught.exception).lower()
        self.assertIn("controller", message)
        _assert_nothing_sent(self, arm)

    def test_willy_allows_a_line_that_does_not_turn(self) -> None:
        """The control: the refusal above is the turn and not the mode."""
        arm = _arm(tool_source="willy")
        self.enterContext(arm.without_camera_world(_DECLINED))
        arm.move_linear(_pose(x=400.0))
        _assert_one_line(self, arm)


class TheVerbsRouteThroughTheCheckedLineTests(unittest.TestCase):
    """`move(linear=True)`, `move_to` and `move_linear` are one line check with three doors."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_move_with_linear_true_runs_the_line(self) -> None:
        arm = _arm()
        self.enterContext(arm.without_camera_world(_DECLINED))
        result = arm.move(_pose(), linear=True)
        self.assertIs(MotionStatus.EXECUTED, result.status)
        self.assertGreater(len(_ik_calls(arm)), 3)

    def test_move_with_linear_true_returns_a_typed_refusal(self) -> None:
        planner = _RecordingPlanner(
            JointCheckVerdict(valid=False, first_invalid=0, checked=9, reason="refused")
        )
        arm = _arm(planner=planner)
        self.enterContext(arm.without_camera_world(_DECLINED))
        result = arm.move(_pose(), linear=True)
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)

    def test_a_line_the_controllers_library_refuses_before_sending_says_so(self) -> None:
        """P0g R3: a value the controller's library refuses before moveL (the transport types it INVALID_TARGET) is a
        refusal before the send, not "the arm may have moved part of the way"."""
        arm = _arm()
        self.enterContext(arm.without_camera_world(_DECLINED))
        _fake(arm).script("move_pose", Rejects(status=MotionStatus.INVALID_TARGET))
        result = arm.move(_pose(), linear=True)
        self.assertIs(MotionStatus.INVALID_TARGET, result.status)
        self.assertIn("before moveL was sent", result.message or "")

    def test_move_without_linear_still_plans(self) -> None:
        """The control: a plain `move` is the cuRobo plan it has always been, not a line."""
        arm = _arm()
        arm._drive_curobo = lambda pose, **kw: MotionResultStub.EXECUTED  # type: ignore[assignment]
        arm.move(_pose())
        self.assertEqual(0, len(_ik_calls(arm)), "a plain move sampled a line")


class ThePolicyDescentIsACheckedLineTests(unittest.TestCase):
    """Step 8p: a pick from the service's policy on a cuRobo arm reaches the planner's check once for the descent and
    once for the lift; the planned standoff reaches the planner, not the check."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_the_policy_descent_and_lift_each_reach_check_joint_path_once(self) -> None:
        from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
        from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

        arm = _arm()
        self.enterContext(arm.without_camera_world(_DECLINED))
        arm._drive_curobo = lambda pose, **kw: MotionResultStub.EXECUTED  # type: ignore[assignment]
        planner = arm._curobo
        planner.detach_payload = lambda: True  # type: ignore[union-attr]
        checks: list[int] = []
        original = planner.check_joint_path  # type: ignore[union-attr]

        def counted(samples, *, refresh=True):  # noqa: ANN001, ANN202
            checks.append(1)
            return original(samples, refresh=refresh)

        planner.check_joint_path = counted  # type: ignore[union-attr, method-assign]
        policy = GraspExecutionPolicy(arm=arm, gripper=None, standoff_mm=30.0, retreat_mm=50.0)
        grasp = GraspPoint(position=np.array([400.0, 0.0, 300.0]), approach=np.array([0.0, 0.0, -1.0]),
                           axis=np.array([-1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                           label="test")

        report = policy.execute(grasp)

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED, report.motion_message)
        self.assertEqual(2, len(checks), "the descent and the lift did not each reach the planner's check once")
        self.assertEqual(2, len(_lines(arm)), "the descent and the lift were not each sent as one moveL")


class MotionResultStub:
    from src.robot.core import MotionCommand, MotionResult

    EXECUTED = MotionResult.executed(MotionCommand.MOVE_TO)


class AnIkArmIsUnchangedTests(unittest.TestCase):
    """The control for the whole file: an ik cell keeps the endpoint gate and one moveL."""

    def test_an_ik_arm_solves_the_endpoint_only_and_sends_one_movel(self) -> None:
        """One IK call, which is the endpoint the preflight context has always needed."""
        arm = _arm(planner_kind="ik")
        arm.move_linear(_pose())
        self.assertEqual(1, len(_ik_calls(arm)), "an ik arm sampled a line")
        _assert_one_line(self, arm)

    def test_the_control_a_curobo_arm_solves_many(self) -> None:
        """Without this the test above would pass on a cell that checks nothing anywhere."""
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")
        arm = _arm()
        self.enterContext(arm.without_camera_world(_DECLINED))
        arm.move_linear(_pose())
        self.assertGreater(len(_ik_calls(arm)), 3)


if __name__ == "__main__":
    unittest.main()
