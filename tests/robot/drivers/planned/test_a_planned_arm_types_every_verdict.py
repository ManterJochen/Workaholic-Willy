"""S/P5: ``move()`` and ``move_home()`` on a planned arm map every safety verdict to a typed MotionStatus, hardware-free.

Measured gap (2026-08, on the UR driver): ``grep "URRobotArm(" tests/`` found exactly one construction (an I/O test
that never calls move()). Yet move() has real September callers -- hand-eye calibration (``execution/calibration.py``)
is the first thing that runs, and ``service.py`` drives it every pick. This pins its contract: frame check, the cuRobo
branch, IK failure, connection faults, all six guard vetoes, the controller's classified refusal passed through, and
the home move boxed and gated like every other joint move.

The real day-one risk this covers is OVER-rejection (a real arm refusing every Cartesian move), not unguarded motion:
the preflight fails CLOSED on missing data, so a bug here reads as a stuck cell, and a stuck cell with no test is how you
lose a bench day.

Moved off ``tests/robot/drivers/ur/test_ur_arm.py`` in P0g slice 13: the verdicts are decided in ``PlannedArm``, so they are held on the
kit's planned arm over the fake transport (``tests/_arm_kit.bare_arm``), and what reached the controller is read off the
fake's tape. What stays in the UR's file is the UR's own: connect's payload push and its gates, and which home its
constructor states.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotConnectionError,
    RobotKinematicsError,
)
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.safety import SafetyDecision, SafetyReason
from src.robot.safety.planning import CuroboUnavailableError
from tests._arm_kit import PlannerDouble, bare_arm, controller_pose, kit_tree, planned_arm
from tests._fake_transport import Rejects, fake_transport
from tests._plan_end import pose_where_it_ends

#: Said where each arm is built: these doubles plan or check with cuRobo and carry no camera world, so every
#: motion they command declines it, as a cuRobo cell must since the world became mandatory.
_DECLINED = "unit double: this test exercises the planner and the guard on a cuRobo arm, and no camera world is wired to it"

_JOINTS = JointPositions([0.0, -1.5, 1.5, 0.0, 1.5, 0.0])


def _pose(frame: Frame = Frame.BASE) -> Pose:
    return Pose(
        position_mm=np.array([400.0, 0.0, 300.0]),
        quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
        frame=frame,
    )


def _fake(arm: PlannedArm) -> Any:
    return arm._transport


class _Preflight:
    """context_for_pose is a passthrough; evaluate() returns a pre-set decision. as_motion_result is a
    class staticmethod, so the REAL reason->status mapping still runs on our crafted decision."""

    def __init__(self, decision: SafetyDecision) -> None:
        self._decision = decision

    def context_for_pose(self, *a: object, **k: object) -> object:
        return object()

    def evaluate(self, ctx: object) -> SafetyDecision:
        return self._decision


def _accept() -> SafetyDecision:
    return SafetyDecision(accepted=True, reason=SafetyReason.OK, guard="test", message="")


def _gated(decision: SafetyDecision) -> Any:
    """An ik arm whose IK answers and whose preflight answers ``decision``, so what a test reads is the mapping."""
    arm: Any = bare_arm("ik")
    arm.ik = lambda pose: _JOINTS
    arm.get_joint_positions = lambda: _JOINTS
    arm._preflight = _Preflight(decision)
    return arm


class FrameAndIkPathTests(unittest.TestCase):
    def test_non_base_frame_is_invalid_target(self) -> None:
        # The frame check is first and needs no connection.
        result = bare_arm().move(_pose(Frame.TOOL))
        self.assertIs(result.status, MotionStatus.INVALID_TARGET)

    def test_ik_failure_is_ik_failed(self) -> None:
        arm: Any = bare_arm("ik")
        arm.ik = lambda pose: (_ for _ in ()).throw(RobotKinematicsError("no solution"))
        self.assertIs(arm.move(_pose()).status, MotionStatus.IK_FAILED)

    def test_ik_connection_error_is_connection_error(self) -> None:
        arm: Any = bare_arm("ik")
        arm.ik = lambda pose: (_ for _ in ()).throw(RobotConnectionError("dropped"))
        self.assertIs(arm.move(_pose()).status, MotionStatus.CONNECTION_ERROR)

    def test_move_to_connection_error_is_connection_error(self) -> None:
        arm = _gated(_accept())
        _fake(arm).script("move_pose", RobotConnectionError("lost mid-move"))
        self.assertIs(arm.move(_pose()).status, MotionStatus.CONNECTION_ERROR)
        self.assertEqual(1, len(_fake(arm).calls("move_pose")), "the fault was not the controller's")


class GuardVetoTests(unittest.TestCase):
    """Every guard's rejection must surface as its typed MotionStatus through move()."""

    _CASES = [
        (SafetyReason.WORKSPACE, MotionStatus.WORKSPACE_REJECTED),
        (SafetyReason.JOINT_LIMIT, MotionStatus.JOINT_LIMIT_REJECTED),
        (SafetyReason.IK_QUALITY, MotionStatus.IK_QUALITY_REJECTED),
        (SafetyReason.SELF_COLLISION, MotionStatus.SELF_COLLISION_REJECTED),
        (SafetyReason.PAYLOAD, MotionStatus.PAYLOAD_REJECTED),
        (SafetyReason.CONTINUITY, MotionStatus.CONTINUITY_REJECTED),
    ]

    def test_each_guard_veto_maps_to_its_status(self) -> None:
        for reason, status in self._CASES:
            with self.subTest(reason=reason):
                arm = _gated(SafetyDecision(accepted=False, reason=reason, guard=reason.value, message="veto"))
                self.assertIs(arm.move(_pose()).status, status)
                # Nothing reached the controller after a veto, on a tape that holds the steady gate asked before it.
                fake = _fake(arm)
                self.assertTrue(fake.calls("wait_until_steady"), "the fake recorded nothing")
                self.assertEqual([], fake.calls("move_pose"), "moved after veto")


class SuccessAndPassthroughTests(unittest.TestCase):
    def test_accept_then_move_success(self) -> None:
        arm = _gated(_accept())
        self.assertTrue(arm.move(_pose()).ok)
        self.assertEqual(1, len(_fake(arm).calls("move_pose")))

    def test_controller_reject_carries_last_reject_status(self) -> None:
        # The controller refuses; the arm must attach the controller's CLASSIFIED cause (a real
        # near-singularity -> IK_QUALITY_REJECTED), not from_bool's generic CONTROLLER_REJECTED.
        arm = _gated(_accept())
        _fake(arm).script("move_pose", Rejects(status=MotionStatus.IK_QUALITY_REJECTED))
        self.assertIs(arm.move(_pose()).status, MotionStatus.IK_QUALITY_REJECTED)

    def test_controller_reject_without_classification_is_generic(self) -> None:
        arm = _gated(_accept())
        _fake(arm).script("move_pose", Rejects(status=None))
        self.assertIs(arm.move(_pose()).status, MotionStatus.CONTROLLER_REJECTED)


class CuroboBranchTests(unittest.TestCase):
    def _arm(self, planner: PlannerDouble | None, *, connected: bool = True) -> Any:
        fake = fake_transport(joints=[0.0, -1.5, 1.5, 0.0, 1.5, 0.0], connected=connected)
        arm: Any = planned_arm(kit_tree("curobo"), transport=fake)
        arm._curobo = planner  # bypass the lazy builder
        return arm

    def test_disconnected_is_connection_error(self) -> None:
        arm = self._arm(None, connected=False)
        self.enterContext(arm.without_camera_world(_DECLINED))
        self.assertIs(arm.move(_pose()).status, MotionStatus.CONNECTION_ERROR)

    def test_curobo_unavailable_is_controller_rejected(self) -> None:
        arm = self._arm(PlannerDouble(plan_result=CuroboUnavailableError("no GPU env")))
        self.enterContext(arm.without_camera_world(_DECLINED))
        self.assertIs(arm.move(_pose()).status, MotionStatus.CONTROLLER_REJECTED)

    def test_no_plan_is_timeout(self) -> None:
        # Empty trajectory -> no collision-free plan -> fail safe, no blind motion.
        arm = self._arm(PlannerDouble(plan_result=[]))
        self.enterContext(arm.without_camera_world(_DECLINED))
        self.assertIs(arm.move(_pose()).status, MotionStatus.TIMEOUT)

    def test_planned_config_gated_before_execute(self) -> None:
        # A plan exists, but the guard rejects cuRobo's ACTUAL final config -> the matching status, and
        # execute() is never reached.
        planner = PlannerDouble(plan_result=[[0.0, -1.5, 1.5, 0.0, 1.5, 0.0]])
        arm = self._arm(planner)
        self.enterContext(arm.without_camera_world(_DECLINED))
        # The goal is where the plan ends: a planned arm refuses a plan off its goal before any gate (Step 8f).
        goal = pose_where_it_ends(arm, [0.0, -1.5, 1.5, 0.0, 1.5, 0.0])
        veto = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, target_pose=goal)
        arm._gate_planned_config = lambda pose, joints: veto
        self.assertIs(arm.move(goal).status, MotionStatus.SELF_COLLISION_REJECTED)

    def test_planned_config_accepted_executes(self) -> None:
        ok = MotionResult.from_bool(True, MotionCommand.MOVE_TO, target_pose=_pose())
        planner = PlannerDouble(plan_result=[[0.0, -1.5, 1.5, 0.0, 1.5, 0.0]], execute_result=ok)
        arm = self._arm(planner)
        self.enterContext(arm.without_camera_world(_DECLINED))
        arm._gate_planned_config = lambda pose, joints: None
        self.assertTrue(arm.move(pose_where_it_ends(arm, [0.0, -1.5, 1.5, 0.0, 1.5, 0.0])).ok)


#: The Isaac cell's frame, approaching along flange +Y, which the home fixtures carry.
_PLUS_Y = {"source": "willy", "offset_mm": (0.0, 132.0, 0.0),
           "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)}


class MoveHomeIsGatedTests(unittest.TestCase):
    """`move_home()` is the natural FIRST motion on a new cell, and it was the only commanded motion on
    the real path that no safety guard ever saw.

    It went straight to `MotionController.move_home`, which checked the workspace box, logged a warning,
    and -- in its own words -- proceeded anyway (P0e deleted it). Measured 2026-08-09 on the September UR3e: the shipped
    UR5e-authored home (then `HOME_JOINTS_DEFAULT`, since P0e each description's `home_rad`) puts the grasp centre at z = 561.9 mm and r = 466.9 mm (93.4%
    of a 500 mm reach) while that cell's `robot.motion.workspace` stops at z = 320. So the first thing an
    operator does would have driven outside the declared box on a log line.

    Two halves, and neither subsumes the other: `gate_joint_target` runs the DESTINATION guards (joint
    limits, self-collision, payload) and deliberately SKIPS the Cartesian workspace box, because that
    guard does not apply to a point-to-point joint command -- but here the box matters, because a home
    pose is a place the arm will sit.

    These arms are `motion_planner: ik` on purpose, which the schema default is not: on a cuRobo
    cell `move_home` also has the whole line to the home configuration judged by the guards and by
    the planner, and that half is pinned in tests/robot/safety/test_a_joint_move_is_judged_as_a_path.py. Here the question
    is the box and the destination guards, so the planner is kept out of the picture. Which home the UR's own
    constructor states stays in tests/robot/drivers/ur/test_ur_arm.py.
    """

    @staticmethod
    def _arm() -> Any:
        fake = fake_transport()
        # FK stands in for the controller: report the pose the caller wants tested.
        fake.always("controller_fk", controller_pose([0.4, 0.0, 0.3, 0.0, 3.14, 0.0]))   # (400, 0, 300) mm, inside
        return planned_arm(kit_tree("ik", safety={}, gripper={"tool_frame": _PLUS_Y}),
                           transport=fake)

    def _assert_not_moved(self, arm: Any) -> None:
        """No joint move, on a tape that holds the home pose read through the controller's FK (the anchor)."""
        fake = _fake(arm)
        self.assertTrue(fake.calls("controller_fk"), "the home pose was never read, so nothing here was boxed")
        self.assertEqual([], fake.calls("run_trajectory"))

    def test_a_home_pose_inside_the_box_still_moves(self) -> None:
        arm = self._arm()
        self.assertTrue(arm.move_home())
        self.assertTrue(_fake(arm).calls("run_trajectory"))

    def test_a_home_pose_outside_the_workspace_is_REFUSED_not_warned(self) -> None:
        """The exact September case: the shipped default lands above a UR3e's z_max."""
        arm = self._arm()
        _fake(arm).always("controller_fk", controller_pose([0.0, -0.22, 0.562, 0.0, 3.14, 0.0]))   # z = 562 mm
        self.assertFalse(arm.move_home())
        self._assert_not_moved(arm)

    def test_a_guard_rejection_stops_it(self) -> None:
        """The destination guards must reach move_home like every other joint move."""
        arm = self._arm()
        arm._preflight = MagicMock()
        arm._preflight.gate_joint_target.return_value = MotionResult.failed(
            MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS, message="test veto",
        )
        self.assertFalse(arm.move_home())
        self.assertEqual([], _fake(arm).calls("run_trajectory"))
        arm._preflight.gate_joint_target.assert_called()

    def test_an_unreadable_home_pose_refuses_rather_than_proceeding(self) -> None:
        """If FK cannot be read, the box cannot be checked -- and an unchecked home move is exactly what
        this guard exists to stop."""
        arm = self._arm()
        _fake(arm).always("controller_fk", RuntimeError("controller dropped"))
        self.assertFalse(arm.move_home())
        self._assert_not_moved(arm)

    def test_the_box_judges_the_grasp_centre_not_the_bare_flange(self) -> None:
        """In willy mode the controller's FK is the bare flange, and `robot.motion.workspace` bounds the grasp
        centre. Here the flange sits at z = 450 mm, inside the default z_max of 500, turned so its +Y
        points up: the grasp centre, 132 mm along flange +Y, is at z = 582 mm and outside."""
        arm = self._arm()
        _fake(arm).always("controller_fk", controller_pose([0.4, 0.0, 0.45, math.pi / 2, 0.0, 0.0]))
        self.assertFalse(arm.move_home(), "move_home boxed the flange instead of the grasp centre")
        self._assert_not_moved(arm)


if __name__ == "__main__":
    unittest.main()
