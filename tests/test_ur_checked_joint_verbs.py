"""On a cuRobo UR, a joint move is a path, and the path is judged before the controller hears of it.

A joint verb on this driver has always been gated at its destination only. That is the right check for
a destination and no check at all for the line the controller runs to get there: `moveJ` interpolates
in joint space, so the arm sweeps through every configuration between where it is and where it was
told to go, and nothing looked at any of them. An endpoint clear of the bench with the bench in the
middle of the swing passes every guard there is.

So on a cuRobo cell the joint verbs now do what the Cartesian one does: sample the line from the
current configuration to the goal, judge every sample with the local guards, ask the planner about the
same samples against the world it holds, and only then send one `moveJ`. Both authorities, because
they see different things: the guards carry the declared fixtures and the exact meshes, the planner
carries the camera world and the attached payload.

An ik cell is untouched, and the control at the bottom of this file is what says so: no planner, no
path, one destination gate and one `moveJ`, exactly as before.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions, MotionStatus
from src.robot.core.errors import RobotConnectionError, RobotMotionRejected
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety.planning import CuroboUnavailableError, JointCheckVerdict

#: Clear, and where the fake controller says the arm is standing.
_HERE = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)
#: Clear, a short move away: about 260 mm of arm travel, so 27 samples at a 10 mm margin.
_THERE = (0.2, -1.5, 1.5, 0.0, 0.0, 0.0)
#: A finger folded into the forearm. The exact mesh backend refuses it.
_FOLDED = (1.95, 0.38, -1.33, -0.55, 2.00, 0.79)


def _mesh_backend_available() -> bool:
    return mesh_backend_status("ur5e") == "ok"


class _RecordingPlanner:
    """Answers `check_joint_path` and records what it was asked."""

    def __init__(self, verdict: JointCheckVerdict | Exception | None = None) -> None:
        self._verdict = verdict
        self.checked: list[list[float]] | None = None

    def check_joint_path(self, samples):
        self.checked = [list(s) for s in samples]
        if isinstance(self._verdict, Exception):
            raise self._verdict
        if self._verdict is None:
            return JointCheckVerdict(
                valid=True, first_invalid=None, checked=len(self.checked), reason="clear"
            )
        return self._verdict


def _arm(
    planner_kind: str = "curobo",
    *,
    connected: bool = True,
    planner: _RecordingPlanner | None = None,
    here: tuple[float, ...] = _HERE,
) -> URRobotArm:
    config = RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"motion_planner": planner_kind},
        "safety": {"payload": {"enforce": False}},
    })
    arm = URRobotArm(config)
    arm._conn = MagicMock()
    arm._conn.is_connected = connected
    arm._conn.get_joint_positions.return_value = list(here)
    arm._conn.moveJ.return_value = True
    # `move_home` boxes the home pose, which it reads through the controller's own FK. A pose
    # inside the shipped workspace box, so the box is not what these tests are measuring.
    arm._conn.fk.return_value = [0.4, 0.0, 0.3, 0.0, 3.14159, 0.0]
    if planner_kind == "curobo":
        arm._curobo_ur = planner if planner is not None else _RecordingPlanner()
    return arm


class TheWholeJointPathIsJudgedTests(unittest.TestCase):
    """The samples reach both authorities, and the controller hears one command."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_the_planner_sees_the_line_from_here_to_the_goal(self) -> None:
        planner = _RecordingPlanner()
        arm = _arm(planner=planner)
        arm.move_joint(JointPositions(_THERE))

        assert planner.checked is not None
        self.assertGreater(len(planner.checked), 2, "the planner saw the endpoints and no line")
        np.testing.assert_allclose(planner.checked[0], _HERE, atol=1e-9)
        np.testing.assert_allclose(planner.checked[-1], _THERE, atol=1e-9)

    def test_one_movej_reaches_the_controller_and_it_is_the_goal(self) -> None:
        arm = _arm()
        arm.move_joint(JointPositions(_THERE))
        self.assertEqual(1, arm._conn.moveJ.call_count)
        np.testing.assert_allclose(arm._conn.moveJ.call_args.args[0], _THERE, atol=1e-9)

    def test_a_guard_refusal_in_the_middle_raises_and_commands_nothing(self) -> None:
        """The endpoint is clear and the middle is not, which is the whole reason for the change."""
        planner = _RecordingPlanner()
        arm = _arm(planner=planner, here=_FOLDED)
        # From the fold to a clear configuration a third of a radian away: the ends are clear and
        # the fold itself is the first sample.
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_joint(JointPositions((1.95, 0.38, -1.33, -0.55, 1.65, 0.79)))
        self.assertIn("sample", str(caught.exception))
        self.assertEqual(0, arm._conn.moveJ.call_count, "a refused path was commanded")
        self.assertIsNone(planner.checked, "the planner was asked about a path the guards refused")

    def test_the_refusal_carries_the_typed_result(self) -> None:
        arm = _arm(here=_FOLDED)
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_joint(JointPositions((1.95, 0.38, -1.33, -0.55, 1.65, 0.79)))
        result = caught.exception.result
        self.assertIsNotNone(result, "the exception carries no result to report")
        assert result is not None
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)

    def test_a_world_refusal_raises_and_commands_nothing(self) -> None:
        planner = _RecordingPlanner(
            JointCheckVerdict(
                valid=False, first_invalid=4, checked=27,
                reason="the cuRobo check refuses sample 4 of 27",
            )
        )
        arm = _arm(planner=planner)
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_joint(JointPositions(_THERE))
        self.assertIn("sample 4 of 27", str(caught.exception))
        self.assertEqual(0, arm._conn.moveJ.call_count)

    def test_an_unavailable_sidecar_raises_and_commands_nothing(self) -> None:
        """Fail closed. A cell configured for cuRobo does not quietly move without it."""
        planner = _RecordingPlanner(CuroboUnavailableError("no GPU env"))
        arm = _arm(planner=planner)
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_joint(JointPositions(_THERE))
        self.assertIn("cuRobo", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, CuroboUnavailableError)
        self.assertEqual(0, arm._conn.moveJ.call_count)

    def test_a_disconnected_arm_says_so_before_any_guard_runs(self) -> None:
        """A path starts at the current joints, and a disconnected arm has none to give."""
        arm = _arm(connected=False)
        with self.assertRaises(RobotConnectionError):
            arm.move_joint(JointPositions(_THERE))
        self.assertEqual(0, arm._conn.moveJ.call_count)


class TheTypedTwinReturnsWhatTheOtherRaisesTests(unittest.TestCase):
    """`move_to_joints` is the typed twin, so every refusal above is a typed result here."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_a_clear_path_executes(self) -> None:
        arm = _arm()
        result = arm.move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.EXECUTED, result.status)
        self.assertEqual(1, arm._conn.moveJ.call_count)

    def test_a_guard_refusal_is_typed(self) -> None:
        arm = _arm(here=_FOLDED)
        result = arm.move_to_joints(JointPositions((1.95, 0.38, -1.33, -0.55, 1.65, 0.79)))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertEqual(0, arm._conn.moveJ.call_count)

    def test_a_world_refusal_is_typed(self) -> None:
        planner = _RecordingPlanner(
            JointCheckVerdict(
                valid=False, first_invalid=1, checked=27, reason="the cuRobo check refuses sample 1"
            )
        )
        result = _arm(planner=planner).move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)

    def test_an_unavailable_sidecar_is_typed(self) -> None:
        planner = _RecordingPlanner(CuroboUnavailableError("no GPU env"))
        result = _arm(planner=planner).move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, result.status)
        self.assertIn("cuRobo", result.message or "")

    def test_a_disconnected_arm_is_typed(self) -> None:
        """Today this raises out of the driver, which a typed verb should never do."""
        result = _arm(connected=False).move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.CONNECTION_ERROR, result.status)


class HomeTakesTheCheckedPathTests(unittest.TestCase):
    """`move_home` is a joint move to a configured configuration, so it is judged like one."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_home_goes_through_the_check_and_sends_one_movej(self) -> None:
        planner = _RecordingPlanner()
        arm = _arm(planner=planner)
        arm._home_joints = list(_THERE)
        self.assertTrue(arm.move_home())
        self.assertIsNotNone(planner.checked)
        self.assertEqual(1, arm._conn.moveJ.call_count)
        np.testing.assert_allclose(arm._conn.moveJ.call_args.args[0], _THERE, atol=1e-9)

    def test_a_refused_path_home_is_false_and_commands_nothing(self) -> None:
        planner = _RecordingPlanner(
            JointCheckVerdict(valid=False, first_invalid=0, checked=27, reason="refused")
        )
        arm = _arm(planner=planner)
        arm._home_joints = list(_THERE)
        self.assertFalse(arm.move_home())
        self.assertEqual(0, arm._conn.moveJ.call_count)

    def test_the_async_twin_runs_the_same_body(self) -> None:
        planner = _RecordingPlanner()
        arm = _arm(planner=planner)
        arm._home_joints = list(_THERE)
        self.assertTrue(asyncio.run(arm.amove_home()))
        self.assertIsNotNone(planner.checked)
        self.assertEqual(1, arm._conn.moveJ.call_count)


class AnIkArmIsUnchangedTests(unittest.TestCase):
    """The control. Without it every test above could be describing a change to every UR cell."""

    def test_an_ik_arm_asks_no_planner_and_sends_one_movej(self) -> None:
        arm = _arm("ik")
        self.assertFalse(hasattr(arm, "_curobo_ur") and arm._curobo_ur is not None)
        arm.move_joint(JointPositions(_THERE))
        self.assertEqual(1, arm._conn.moveJ.call_count)

    def test_an_ik_arm_still_refuses_a_bad_destination(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")
        arm = _arm("ik")
        with self.assertRaises(RobotMotionRejected):
            arm.move_joint(JointPositions(_FOLDED))
        self.assertEqual(0, arm._conn.moveJ.call_count)

    def test_an_ik_arm_moves_through_a_configuration_nobody_judged(self) -> None:
        """The honest statement of what an ik cell still is: the destination, and nothing between.

        This is not a defect this step fixes. It is the reason the cuRobo cells above changed, and it
        is written down here so an ik cell is never read as one that got the same check.
        """
        arm = _arm("ik", here=_FOLDED)
        arm.move_joint(JointPositions(_THERE))
        self.assertEqual(1, arm._conn.moveJ.call_count, "the swing out of the fold was commanded")


if __name__ == "__main__":
    unittest.main()
