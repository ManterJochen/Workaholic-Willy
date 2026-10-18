"""On a cuRobo arm, a joint move is a path, and the path is judged before the controller hears of it.

A joint verb on the UR driver had always been gated at its destination only. That is the right check for
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

The arm is a ``PlannedArm`` over the fake transport since P0g slice 13 (``tests/_arm_kit.planned_arm``), so a joint
move is read as a ``run_trajectory`` on the fake's tape, and every "nothing was sent" first checks that the tape
holds where the arm was read to stand (``tests/repo/test_the_kit_tests_read_what_the_fake_records.py``).
"""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.robot.core import JointPositions, MotionStatus
from src.robot.core.errors import RobotConnectionError, RobotMotionRejected
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety.planning import CuroboUnavailableError, JointCheckVerdict
from tests._arm_kit import RecordingPlanner, controller_pose, kit_tree, planned_arm
from tests._fake_transport import Raises, fake_transport

#: Said where each arm is built: these doubles plan or check with cuRobo and carry no camera world, so every
#: motion they command declines it, as a cuRobo cell must since the world became mandatory.
_DECLINED = "unit double: this test exercises the planner and the guard on a cuRobo arm, and no camera world is wired to it"

#: Clear, and where the fake controller says the arm is standing.
_HERE = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)
#: Clear, a short move away: about 295 mm of travel with the 2F-85 counted in the base joint's radius,
#: so 30 samples at a 10 mm margin.
_THERE = (0.2, -1.5, 1.5, 0.0, 0.0, 0.0)
#: A finger folded into the forearm. The exact mesh backend refuses it.
_FOLDED = (1.95, 0.38, -1.33, -0.55, 2.00, 0.79)


def _mesh_backend_available() -> bool:
    return mesh_backend_status("ur5e") == "ok"


#: The recording planner, held by the kit since P0g slice 13.
_RecordingPlanner = RecordingPlanner


def _arm(
    planner_kind: str = "curobo",
    *,
    connected: bool = True,
    planner: RecordingPlanner | None = None,
    here: tuple[float, ...] = _HERE,
    home: Sequence[float] | None = None,
) -> PlannedArm:
    fake = fake_transport(joints=here, connected=connected)
    # `move_home` boxes the home pose, which it reads through the controller's own FK. A pose
    # inside the shipped workspace box, so the box is not what these tests are measuring.
    fake.always("controller_fk", controller_pose([0.4, 0.0, 0.3, 0.0, 3.14159, 0.0]))
    arm: Any = planned_arm(kit_tree(planner_kind, safety={}), transport=fake,
                           home_joints=home)
    if planner_kind == "curobo":
        arm._curobo = planner if planner is not None else _RecordingPlanner()
    return arm  # type: ignore[no-any-return]


def _moves(arm: PlannedArm) -> list[Any]:
    """Every joint move the arm sent: one ``run_trajectory`` each, its waypoints the first argument."""
    fake: Any = arm._transport
    return fake.calls("run_trajectory")  # type: ignore[no-any-return]


def _assert_nothing_sent(test: unittest.TestCase, arm: PlannedArm, msg: str | None = None) -> None:
    """No joint move, on a tape that holds where the arm was read to stand (the anchor)."""
    fake: Any = arm._transport
    test.assertTrue(fake.calls("read_joints"), "the fake recorded no read, so 'nothing was sent' would read nothing")
    test.assertEqual([], _moves(arm), msg)


def _assert_one_move_to(test: unittest.TestCase, arm: PlannedArm, goal: Sequence[float], msg: str | None = None) -> None:
    moves = _moves(arm)
    test.assertEqual(1, len(moves), msg)
    waypoints = moves[0].args[0]
    test.assertEqual(1, len(waypoints), "one moveJ is one waypoint")
    np.testing.assert_allclose(waypoints[0], goal, atol=1e-9)


class TheWholeJointPathIsJudgedTests(unittest.TestCase):
    """The samples reach both authorities, and the controller hears one command."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_the_planner_sees_the_line_from_here_to_the_goal(self) -> None:
        planner = _RecordingPlanner()
        arm = _arm(planner=planner)
        self.enterContext(arm.without_camera_world(_DECLINED))
        self.assertTrue(arm.move_to_joints(JointPositions(_THERE)).ok)

        assert planner.checked is not None
        self.assertGreater(len(planner.checked), 2, "the planner saw the endpoints and no line")
        np.testing.assert_allclose(planner.checked[0], _HERE, atol=1e-9)
        np.testing.assert_allclose(planner.checked[-1], _THERE, atol=1e-9)
        self.assertEqual(3.0, planner.clearance_mm, "the line was not judged at path_samples.LINE_CLEARANCE_MM")

    def test_one_movej_reaches_the_controller_and_it_is_the_goal(self) -> None:
        arm = _arm()
        self.enterContext(arm.without_camera_world(_DECLINED))
        self.assertTrue(arm.move_to_joints(JointPositions(_THERE)).ok)
        _assert_one_move_to(self, arm, _THERE)


class EveryRefusalIsATypedResultTests(unittest.TestCase):
    """Every refusal of the joint verb is a typed result, and a refused path commands nothing."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_a_clear_path_executes(self) -> None:
        arm = _arm()
        self.enterContext(arm.without_camera_world(_DECLINED))
        result = arm.move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.EXECUTED, result.status)
        self.assertEqual(1, len(_moves(arm)))

    def test_a_guard_refusal_in_the_middle_is_typed_and_commands_nothing(self) -> None:
        """The endpoint is clear and the middle is not, which is the whole reason for the change."""
        planner = _RecordingPlanner()
        arm = _arm(planner=planner, here=_FOLDED)
        self.enterContext(arm.without_camera_world(_DECLINED))
        # From the fold to a clear configuration a third of a radian away: the ends are clear and
        # the fold itself is the first sample.
        result = arm.move_to_joints(JointPositions((1.95, 0.38, -1.33, -0.55, 1.65, 0.79)))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertIn("sample", result.message or "")
        _assert_nothing_sent(self, arm, "a refused path was commanded")
        self.assertIsNone(planner.checked, "the planner was asked about a path the guards refused")

    def test_a_world_refusal_is_typed_and_commands_nothing(self) -> None:
        planner = _RecordingPlanner(
            JointCheckVerdict(
                valid=False, first_invalid=4, checked=27,
                reason="the cuRobo check refuses sample 4 of 27",
            )
        )
        arm = _arm(planner=planner)
        self.enterContext(arm.without_camera_world(_DECLINED))
        result = arm.move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertIn("sample 4 of 27", result.message or "")
        _assert_nothing_sent(self, arm)

    def test_an_unavailable_sidecar_is_typed_and_commands_nothing(self) -> None:
        """Fail closed. A cell configured for cuRobo does not quietly move without it."""
        planner = _RecordingPlanner(CuroboUnavailableError("no GPU env"))
        arm = _arm(planner=planner)
        self.enterContext(arm.without_camera_world(_DECLINED))
        result = arm.move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, result.status)
        self.assertIn("cuRobo", result.message or "")
        self.assertIsInstance(result.exception, CuroboUnavailableError)
        _assert_nothing_sent(self, arm)

    def test_a_disconnected_arm_says_so_before_any_guard_runs(self) -> None:
        """A path starts at the current joints, and a disconnected arm has none to give: it asks the controller
        nothing at all, so its tape is empty (the connected tests above are the control that the tape records)."""
        arm = _arm(connected=False)
        self.enterContext(arm.without_camera_world(_DECLINED))
        result = arm.move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.CONNECTION_ERROR, result.status)
        fake: Any = arm._transport
        self.assertEqual([], fake.tape)

    def test_raise_unless_executed_raises_what_the_removed_verb_raised(self) -> None:
        """A refused path raises the rejection with its typed result; a lost link raises the connection error."""
        arm = _arm(here=_FOLDED)
        self.enterContext(arm.without_camera_world(_DECLINED))
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_to_joints(JointPositions((1.95, 0.38, -1.33, -0.55, 1.65, 0.79))).raise_unless_executed()
        assert caught.exception.result is not None
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, caught.exception.result.status)
        gone = _arm(connected=False)
        self.enterContext(gone.without_camera_world(_DECLINED))
        with self.assertRaises(RobotConnectionError):
            gone.move_to_joints(JointPositions(_THERE)).raise_unless_executed()


class HomeTakesTheCheckedPathTests(unittest.TestCase):
    """`move_home` is a joint move to a configured configuration, so it is judged like one."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_home_goes_through_the_check_and_sends_one_movej(self) -> None:
        planner = _RecordingPlanner()
        arm = _arm(planner=planner, home=_THERE)
        self.enterContext(arm.without_camera_world(_DECLINED))
        self.assertTrue(arm.move_home())
        self.assertIsNotNone(planner.checked)
        _assert_one_move_to(self, arm, _THERE)

    def test_a_refused_path_home_is_false_and_commands_nothing(self) -> None:
        """Refused by the planner's verdict. Until P0g slice 13 this test declined no camera world, so its home was
        refused before planning for the missing world and the verdict was never asked; the planner's record says it
        was asked now."""
        planner = _RecordingPlanner(
            JointCheckVerdict(valid=False, first_invalid=0, checked=27, reason="refused")
        )
        arm = _arm(planner=planner, home=_THERE)
        self.enterContext(arm.without_camera_world(_DECLINED))
        self.assertFalse(arm.move_home())
        self.assertIsNotNone(planner.checked, "the home was refused before the planner was asked")
        _assert_nothing_sent(self, arm)


class AnIkArmIsUnchangedTests(unittest.TestCase):
    """The control. Without it every test above could be describing a change to every cell."""

    def test_an_ik_arm_asks_no_planner_and_sends_one_movej(self) -> None:
        arm = _arm("ik")
        self.assertFalse(hasattr(arm, "_curobo") and arm._curobo is not None)
        self.assertTrue(arm.move_to_joints(JointPositions(_THERE)).ok)
        _assert_one_move_to(self, arm, _THERE)

    def test_an_ik_arm_still_refuses_a_bad_destination(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")
        arm = _arm("ik")
        self.assertFalse(arm.move_to_joints(JointPositions(_FOLDED)).ok)
        self.assertEqual([], _moves(arm))

    def test_an_ik_arm_moves_through_a_configuration_nobody_judged(self) -> None:
        """The honest statement of what an ik cell still is: the destination, and nothing between.

        This is not a defect this step fixes. It is the reason the cuRobo cells above changed, and it
        is written down here so an ik cell is never read as one that got the same check.
        """
        arm = _arm("ik", here=_FOLDED)
        self.assertTrue(arm.move_to_joints(JointPositions(_THERE)).ok)
        _assert_one_move_to(self, arm, _THERE, "the swing out of the fold was commanded")


class ARangeErrorIsATypedResultTests(unittest.TestCase):
    """⛔ A VALUE THE CONTROLLER'S LIBRARY REFUSES BEFORE SENDING IS A RESULT, NOT A RAISE (P0g R3, review of the prod
    port 2026-09-24).

    ur_rtde raises ValueError for a speed or an acceleration outside its range before it sends anything, and a
    negative or NaN speed passes the clamp, which only lowers what is above the maximum. The cuRobo execute already
    typed that as INVALID_TARGET; the joint verb let it out of `move_to_joints` as a raw ValueError. A transport
    raises that ValueError for a waypoint it did not send (``ArmTransport``'s contract), which the fake keeps."""

    def test_a_joint_move_the_controllers_library_refuses_says_nothing_moved(self) -> None:
        arm = _arm("ik")
        fake: Any = arm._transport
        fake.script("run_trajectory", Raises(ValueError("The value is not within [0;3.14]")))
        result = arm.move_to_joints(JointPositions(_THERE), velocity=-0.1)
        self.assertIs(MotionStatus.INVALID_TARGET, result.status)
        self.assertIn("moveJ was not sent", result.message or "")
        self.assertIn("nothing moved", result.message or "")
        self.assertEqual(1, len(_moves(arm)), "the refusal was not the controller library's")
        self.assertEqual([], fake.sent, "a waypoint the library refused was taken")


if __name__ == "__main__":
    unittest.main()
