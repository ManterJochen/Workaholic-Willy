"""What a test injects on a UR arm is what the arm reads (P0e, plan section 5.3).

`tests/_injection_guard.py` refuses a write to a name the arm does not carry. It cannot see a name that still exists
but has stopped being the one a code path reads: once the RTDE calls go behind a transport (E7), a transport built from
`_conn` at construction would leave `arm._conn = MagicMock()` replacing a reference nothing uses, and every
`conn.moveJ.assert_not_called()` after it would pass on nothing. Each test here injects one of the names the suite
injects most (counted 2026-09-22: `_conn` 47 times, `_preflight` 20, `_curobo` 18, `ik` 14, `_motion` 6,
`_home_joints` 5, `_motion.conn` 3) and shows a verb reaching it, so the extraction has to keep them live, not only
present. The last test is the guard's own control.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.motion import MotionController
from src.robot.drivers.ur.pose import URPose
from src.robot.safety import SafetyDecision, SafetyReason
from tests._ur_transport import over

_JOINTS = [0.11, -1.52, 1.43, -0.04, 1.55, 0.26]
_HOME = [0.21, -1.32, 1.23, -0.14, 1.45, 0.36]


def _arm(planner: str = "ik") -> URRobotArm:
    return URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "motion": {"planner": planner}, "ur": {"ip": "127.0.0.1"},
        "gripper": {"model": "robotiq_2f85"},
    }))


def _connection() -> MagicMock:
    conn = MagicMock()
    conn.is_connected = True
    conn.get_joint_positions.return_value = list(_JOINTS)
    conn.fk.return_value = [0.3, 0.0, 0.3, 0.0, 3.14159, 0.0]
    conn.moveJ.return_value = True
    conn.wait_until_steady.return_value = True
    return conn


def _pose() -> Pose:
    return Pose(position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                frame=Frame.BASE)


class _Preflight:
    """Answers every question with the decision it was given; ``as_motion_result`` stays the real class's."""

    def __init__(self, decision: SafetyDecision) -> None:
        self.decision = decision
        self.asked: list[str] = []

    def context_for_pose(self, *args: object, **kwargs: object) -> object:
        self.asked.append("context_for_pose")
        return object()

    def evaluate(self, ctx: object) -> SafetyDecision:
        self.asked.append("evaluate")
        return self.decision

    def gate_joint_target(self, joints: object, *, arm: object) -> None:
        self.asked.append("gate_joint_target")
        return None


def _accept() -> SafetyDecision:
    return SafetyDecision(accepted=True, reason=SafetyReason.OK, guard="test", message="")


class WhatATestInjectsReachesTheArmTests(unittest.TestCase):
    def test_the_connection_it_injects_is_the_one_every_read_and_stop_goes_to(self) -> None:
        arm, conn = _arm(), _connection()
        arm._conn = conn
        self.assertIs(arm.is_connected, True)
        self.assertEqual(list(arm.get_joint_positions()), _JOINTS)
        self.assertIs(arm.wait_until_steady(1.5, 0.05), True)
        conn.wait_until_steady.assert_called_once_with(1.5, 0.05)
        # Willy's stop() sends nothing itself: it latches the halt on the connection (SupportsHalt), so the halt
        # request is what reaches the injected one.
        arm.stop()
        conn.request_halt.assert_called_once_with("stop() was called")

    def test_the_motion_controller_it_injects_reports_the_pose_and_runs_the_move(self) -> None:
        arm = _arm()
        arm._conn = _connection()
        motion = MagicMock()
        motion.get_current_pose.return_value = URPose(x=410.0, y=-5.0, z=290.0, rx=0.0, ry=3.14159, rz=0.0)
        motion.move_to.return_value = True
        motion.last_reject_status = None
        arm._motion = motion
        np.testing.assert_allclose(arm.get_tcp_pose().position_mm, [410.0, -5.0, 290.0], atol=1e-9)
        arm.ik = lambda pose, *, seed=None: JointPositions(_JOINTS)  # type: ignore[method-assign]
        arm._preflight = _Preflight(_accept())  # type: ignore[assignment]
        self.assertIs(arm.move(_pose()).status, MotionStatus.EXECUTED)
        motion.move_to.assert_called_once()

    def test_the_preflight_it_injects_is_the_one_that_decides(self) -> None:
        arm = _arm()
        arm._conn = _connection()
        arm.ik = lambda pose, *, seed=None: JointPositions(_JOINTS)  # type: ignore[method-assign]
        preflight = _Preflight(SafetyDecision(accepted=False, reason=SafetyReason.WORKSPACE, guard="test",
                                              message="injected refusal"))
        arm._preflight = preflight  # type: ignore[assignment]
        self.assertIs(arm.safety_preflight, preflight)
        self.assertIs(arm.move(_pose()).status, MotionStatus.WORKSPACE_REJECTED)
        self.assertEqual(preflight.asked, ["context_for_pose", "evaluate"])
        arm._conn.moveJ.assert_not_called()

    def test_the_ik_it_injects_is_the_one_the_move_pre_resolves_with(self) -> None:
        arm = _arm()
        arm._conn = _connection()
        asked: list[Pose] = []

        def ik(pose: Pose, *, seed: JointPositions | None = None) -> JointPositions:
            asked.append(pose)
            return JointPositions(_JOINTS)

        arm.ik = ik  # type: ignore[method-assign]
        arm._preflight = _Preflight(SafetyDecision(accepted=False, reason=SafetyReason.WORKSPACE,  # type: ignore[assignment]
                                                   guard="test", message="stop after the ik"))
        arm.move(_pose())
        self.assertEqual(len(asked), 1)
        np.testing.assert_allclose(asked[0].position_mm, _pose().position_mm)

    def test_the_home_it_injects_is_the_home_that_is_checked_and_driven(self) -> None:
        arm = _arm()
        arm._conn = _connection()
        arm._preflight = _Preflight(_accept())  # type: ignore[assignment]
        arm._home_joints = list(_HOME)
        self.assertIs(arm.move_home(), True)
        self.assertEqual(arm._conn.fk.call_args.args[0], _HOME)
        np.testing.assert_allclose(arm._conn.moveJ.call_args.args[0], _HOME)

    def test_the_planner_it_injects_is_the_one_asked_and_closed(self) -> None:
        arm = _arm("curobo")
        planner = MagicMock()
        arm._curobo = planner
        self.assertIs(arm._curobo_planner(), planner)
        # Willy retires a planner rather than closing it, so a move still holding it starts no sidecar on it.
        arm.stop_planner()
        planner.retire.assert_called_once_with()
        self.assertIsNone(arm._curobo)

    def test_the_connection_injected_on_the_motion_controller_is_the_one_it_moves(self) -> None:
        built, injected = _connection(), _connection()
        injected.ik.return_value = list(_JOINTS)
        guard = MagicMock()
        guard.is_inside_workspace.return_value = True
        controller = MotionController(built, guard, max_velocity=1.0, max_acceleration=1.0)
        controller._is_singularity_risky = lambda joints: False  # type: ignore[method-assign]
        controller.conn = injected
        pose = URPose(x=400.0, y=0.0, z=300.0, rx=0.0, ry=3.14159, rz=0.0, label="t")
        self.assertIs(controller.move_to(pose), True)
        injected.moveJ.assert_called_once()
        built.moveJ.assert_not_called()
        built.ik.assert_not_called()

    def test_the_guard_refuses_a_name_the_arm_does_not_carry(self) -> None:
        """The control for the other half: a misspelt injection fails loudly instead of controlling nothing, a
        present name is accepted, and the library itself stays free to set a new attribute."""
        arm = _arm()
        with self.assertRaisesRegex(AttributeError, r"URRobotArm has no '_connection'"):
            arm._connection = _connection()  # type: ignore[attr-defined]
        with self.assertRaisesRegex(AttributeError, r"MotionController has no 'connection'"):
            arm._motion.connection = _connection()  # type: ignore[attr-defined]
        arm._conn = _connection()
        arm._lift_judged = None  # carried since the build
        hand_made = URRobotArm.__new__(URRobotArm)
        hand_made._transport = over(_connection())  # nothing about it was built, so nothing is refused


if __name__ == "__main__":
    unittest.main()
