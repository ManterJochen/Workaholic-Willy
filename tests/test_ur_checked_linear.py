"""On a cuRobo UR, a straight line is checked where it is straight, one sample at a time.

`moveL` is the one motion this stack does not plan. The controller runs it, and what it runs is a
straight line of the frame IT holds: in `willy` mode a bare flange, in `polyscope` mode the declared
tool. Nothing in this repository has ever looked at the configurations that line passes through, and a
line whose two ends are clear can still sweep the elbow through a bin wall.

So on a cuRobo cell the line is sampled where the controller draws it, every sample is solved by the
controller's own inverse kinematics seeded with the previous solution (the same kinematics `moveL`
uses, so the configurations judged are the ones executed), and those configurations go to the guards
and to the planner before one `moveL` is sent.

Two refusals belong to the line and to nothing else. A jump between neighbouring solutions means the
controller changed branch mid line and the arm will take a route nobody sampled. And in `willy` mode an
orientation change means the grasp centre does NOT travel straight even though the flange does, which
is the owner's answer Q8: refuse above half a degree and say that polyscope mode gives a straight tool
line.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.core.errors import RobotMotionRejected
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.pose import URPose
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety.planning import JointCheckVerdict

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


class _RecordingPlanner:
    def __init__(self, verdict: JointCheckVerdict | Exception | None = None) -> None:
        self._verdict = verdict
        self.checked: list[list[float]] | None = None

    def check_joint_path(self, samples, *, refresh=True):
        self.checked = [list(s) for s in samples]
        if isinstance(self._verdict, Exception):
            raise self._verdict
        if self._verdict is None:
            return JointCheckVerdict(
                valid=True, first_invalid=None, checked=len(self.checked), reason="clear"
            )
        return self._verdict


def _arm(
    *,
    planner_kind: str = "curobo",
    tool_source: str = "polyscope",
    planner: _RecordingPlanner | None = None,
    ik_solutions=None,
) -> URRobotArm:
    """A UR whose controller answers IK with `ik_solutions`, one per call, then repeats the last."""
    config = RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"motion_planner": planner_kind},
        # The guards this file is not about: the fake controller answers every sample with the
        # same clear configuration, whose Jacobian a real cell would never see.
        "safety": {"payload": {"enforce": False}, "ik_quality": {"enforce": False}},
        # The frame of the one hand model this tree holds, approaching along flange +Y: a UR declaring the real
        # flange's +Z refuses to build until a model of that hand exists (Step 4g). The line tests read the source
        # and the turn, not the axis the offset runs along.
        "gripper": {"model": "robotiq_2f85", "tool_frame": {
            "source": tool_source,
            "offset_mm": (0.0, 132.0, 0.0),
            "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
        }},
    })
    arm = URRobotArm(config)
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(_HERE_JOINTS)
    arm._conn.fk_current.return_value = [0.3, 0.0, 0.3, 0.0, 3.14159, 0.0]
    arm._conn.fk.return_value = [0.3, 0.0, 0.3, 0.0, 3.14159, 0.0]
    arm._conn.moveL.return_value = True
    arm._conn.move_to.return_value = True

    solutions = list(ik_solutions) if ik_solutions is not None else None
    arm.ik_calls: list[tuple[Pose, JointPositions | None]] = []  # type: ignore[attr-defined]

    def fake_ik(pose, *, seed=None):
        arm.ik_calls.append((pose, seed))  # type: ignore[attr-defined]
        if solutions is None:
            return JointPositions(_SOLUTION)
        index = min(len(arm.ik_calls) - 1, len(solutions) - 1)  # type: ignore[attr-defined]
        return JointPositions(solutions[index])

    arm.ik = fake_ik  # type: ignore[method-assign]
    arm._motion = MagicMock()
    arm._motion.move_to.return_value = True
    # Where the arm is standing, in the frame the controller reports: 300 mm out, 300 mm up, tool down. In polyscope
    # mode the controller holds the tool and reports the grasp centre's rotation, (0, pi, 0); in willy mode it runs a
    # bare flange, which the declared frame's -90 degrees about X turns back to tool down, so it reports the flange at
    # (0, pi/sqrt(2), -pi/sqrt(2)). Both put the tool where every goal below starts, so a line that keeps (0, pi, 0)
    # does not turn.
    rotation = (0.0, 3.14159265, 0.0) if tool_source == "polyscope" else (0.0, 2.221441469079183, -2.221441469079183)
    arm._motion.get_current_pose.return_value = URPose.from_ur_list([0.3, 0.0, 0.3, *rotation])
    if planner_kind == "curobo":
        arm._curobo_ur = planner if planner is not None else _RecordingPlanner()
    return arm


class TheLineIsSampledAndSolvedTests(unittest.TestCase):
    """Every sample is one seeded IK call, and the solutions are what the guards see."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_one_seeded_ik_call_per_sample_in_order(self) -> None:
        arm = _arm()
        arm.move_linear(_pose(x=400.0))
        calls = arm.ik_calls  # type: ignore[attr-defined]
        self.assertGreater(len(calls), 3, "the line was solved at its ends and nowhere between")
        self.assertIsNone(calls[0][1], "the first sample was seeded with something")
        for index in range(1, len(calls)):
            with self.subTest(sample=index):
                seed = calls[index][1]
                self.assertIsNotNone(seed, "a sample was solved without the previous solution")
                assert seed is not None
                np.testing.assert_allclose(seed.tolist(), _SOLUTION, atol=1e-9)

    def test_the_planner_sees_the_solved_configurations(self) -> None:
        planner = _RecordingPlanner()
        arm = _arm(planner=planner)
        arm.move_linear(_pose())
        assert planner.checked is not None
        self.assertEqual(len(arm.ik_calls), len(planner.checked))  # type: ignore[attr-defined]

    def test_exactly_one_movel_and_no_movej(self) -> None:
        arm = _arm()
        arm.move_linear(_pose())
        self.assertEqual(1, arm._motion.move_to.call_count)
        self.assertTrue(arm._motion.move_to.call_args.kwargs.get("linear"))
        self.assertEqual(0, arm._conn.moveJ.call_count)

    def test_a_branch_jump_between_neighbours_refuses_before_anything_moves(self) -> None:
        """The controller changing IK branch mid line is a route nobody sampled."""
        jump = list(_SOLUTION)
        jump[0] += 2.5
        arm = _arm(ik_solutions=[_SOLUTION, _SOLUTION, tuple(jump), _SOLUTION])
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose())
        self.assertIn("branch", str(caught.exception).lower())
        self.assertEqual(0, arm._motion.move_to.call_count)

    def test_a_planner_refusal_stops_the_line(self) -> None:
        planner = _RecordingPlanner(
            JointCheckVerdict(valid=False, first_invalid=2, checked=9, reason="sample 2 of 9")
        )
        arm = _arm(planner=planner)
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose())
        self.assertIn("sample 2 of 9", str(caught.exception))
        self.assertEqual(0, arm._motion.move_to.call_count)


class TheLineIsSampledWhereTheControllerDrawsItTests(unittest.TestCase):
    """`willy` runs a bare flange, `polyscope` runs the declared tool, and they are different lines."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_polyscope_samples_the_tool_line(self) -> None:
        arm = _arm(tool_source="polyscope")
        arm.move_linear(_pose(x=500.0))
        poses = [call[0] for call in arm.ik_calls]  # type: ignore[attr-defined]
        xs = [float(p.position_mm[0]) for p in poses]
        self.assertAlmostEqual(500.0, xs[-1], places=6, msg="the line does not end at the goal")
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(xs, xs[1:])), "the line doubles back")

    def test_willy_refuses_a_turn_because_the_grasp_centre_would_not_travel_straight(self) -> None:
        """The owner's answer Q8. A straight flange line with a turn is a curved tool line."""
        turned = np.array([0.0, 0.9659258, 0.0, 0.2588190])  # 30 degrees about Y from the start
        arm = _arm(tool_source="willy")
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_linear(_pose(quat=turned))
        message = str(caught.exception).lower()
        self.assertIn("polyscope", message)
        self.assertEqual(0, arm._motion.move_to.call_count)

    def test_willy_allows_a_line_that_does_not_turn(self) -> None:
        """The control: the refusal above is the turn and not the mode."""
        arm = _arm(tool_source="willy")
        arm.move_linear(_pose(x=400.0))
        self.assertEqual(1, arm._motion.move_to.call_count)


class TheVerbsRouteThroughTheCheckedLineTests(unittest.TestCase):
    """`move(linear=True)`, `move_to` and `move_linear` are one line check with three doors."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_move_with_linear_true_runs_the_line(self) -> None:
        arm = _arm()
        result = arm.move(_pose(), linear=True)
        self.assertIs(MotionStatus.EXECUTED, result.status)
        self.assertGreater(len(arm.ik_calls), 3)  # type: ignore[attr-defined]

    def test_move_with_linear_true_returns_a_typed_refusal(self) -> None:
        planner = _RecordingPlanner(
            JointCheckVerdict(valid=False, first_invalid=0, checked=9, reason="refused")
        )
        result = _arm(planner=planner).move(_pose(), linear=True)
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)

    def test_move_without_linear_still_plans(self) -> None:
        """The control: a plain `move` is the cuRobo plan it has always been, not a line."""
        arm = _arm()
        arm._drive_curobo = lambda pose, **kw: MotionResultStub.EXECUTED  # type: ignore[assignment]
        arm.move(_pose())
        self.assertEqual(0, len(arm.ik_calls), "a plain move sampled a line")  # type: ignore[attr-defined]


    def test_move_to_on_a_curobo_cell_reaches_the_planner(self) -> None:
        """It used to command the controller directly and never consult the planner at all."""
        planner = _RecordingPlanner()
        arm = _arm(planner=planner)
        self.assertTrue(arm.move_to(_pose(), linear=True))
        self.assertIsNotNone(planner.checked, "move_to never asked the planner")

    def test_move_to_returns_false_when_the_line_is_refused(self) -> None:
        planner = _RecordingPlanner(
            JointCheckVerdict(valid=False, first_invalid=0, checked=9, reason="refused")
        )
        arm = _arm(planner=planner)
        self.assertFalse(arm.move_to(_pose(), linear=True))
        self.assertEqual(0, arm._motion.move_to.call_count)

    def test_the_async_twin_runs_the_same_body(self) -> None:
        import asyncio

        planner = _RecordingPlanner()
        arm = _arm(planner=planner)
        self.assertTrue(asyncio.run(arm.amove_to(_pose(), linear=True)))
        self.assertIsNotNone(planner.checked)


class MotionResultStub:
    from src.robot.core import MotionCommand, MotionResult

    EXECUTED = MotionResult.executed(MotionCommand.MOVE_TO)


class AnIkArmIsUnchangedTests(unittest.TestCase):
    """The control for the whole file: an ik cell keeps the endpoint gate and one moveL."""

    def test_an_ik_arm_solves_the_endpoint_only_and_sends_one_movel(self) -> None:
        """One IK call, which is the endpoint the preflight context has always needed."""
        arm = _arm(planner_kind="ik")
        arm.move_linear(_pose())
        self.assertEqual(1, len(arm.ik_calls), "an ik arm sampled a line")  # type: ignore[attr-defined]
        self.assertEqual(1, arm._motion.move_to.call_count)

    def test_the_control_a_curobo_arm_solves_many(self) -> None:
        """Without this the test above would pass on a cell that checks nothing anywhere."""
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")
        arm = _arm()
        arm.move_linear(_pose())
        self.assertGreater(len(arm.ik_calls), 3)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
