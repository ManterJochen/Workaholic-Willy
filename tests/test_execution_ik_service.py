"""Tests for the vendor-neutral IK service adapters.

These cover the contract that ``GraspCalculator`` relies on when an
``IKService`` is plugged in: every adapter must turn the underlying
arm's behaviour into an :class:`IKResult`, never propagate
vendor-specific exceptions, and never lie about reachability.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    JointPositions,
    RobotConnectionError,
    RobotKinematicsError,
)
from src.robot.execution.ik_service import (
    CachedIKService,
    RobotArmIKService,
    URAnalyticIKService,
)
from src.robot.grasping.planning.grasp_pose import GraspPose


def _grasp(x: float = 0.0, *, frame: Frame = Frame.BASE) -> GraspPose:
    return GraspPose(
        position_mm=np.array([x, 0.0, 200.0], dtype=np.float64),
        rotation_matrix=np.eye(3, dtype=np.float64),
        grip_width_mm=40.0,
        score=0.5,
        confidence=0.5,
        contacts=(
            np.array([x, -20.0, 200.0], dtype=np.float64),
            np.array([x, 20.0, 200.0], dtype=np.float64),
        ),
        frame=frame,
        metadata={},
    )


class _FakeArm:
    """Minimal arm satisfying the subset of ``RobotArm`` IKService uses."""

    def __init__(
        self,
        *,
        connected: bool = True,
        unreachable_x: float | None = None,
        always_raise_kinematics: bool = False,
    ) -> None:
        self.connected = connected
        self.unreachable_x = unreachable_x
        self.always_raise_kinematics = always_raise_kinematics
        self.ik_calls: list[Pose] = []
        self.seed_requests = 0

    def get_joint_positions(self) -> JointPositions:
        self.seed_requests += 1
        if not self.connected:
            raise RobotConnectionError("not connected")
        return JointPositions([0.0] * 6)

    def ik(self, pose: Pose, *, seed: JointPositions | None = None) -> JointPositions:
        self.ik_calls.append(pose)
        if not self.connected:
            raise RobotConnectionError("not connected")
        if self.always_raise_kinematics:
            raise RobotKinematicsError("no solution")
        if (
            self.unreachable_x is not None
            and abs(pose.position_mm[0] - self.unreachable_x) < 1e-6
        ):
            raise RobotKinematicsError("outside reach")
        return JointPositions([0.1] * 6)


class RobotArmIKServiceTests(unittest.TestCase):
    def test_reachable_pose_returns_joints(self) -> None:
        service = RobotArmIKService(_FakeArm())
        result = service.query(_grasp())
        self.assertTrue(result.reachable)
        self.assertIsNotNone(result.joints)
        self.assertEqual(len(result.joints), 6)

    def test_kinematics_error_maps_to_unreachable(self) -> None:
        service = RobotArmIKService(_FakeArm(always_raise_kinematics=True))
        result = service.query(_grasp())
        self.assertFalse(result.reachable)
        self.assertEqual(result.reason, "ik_no_solution")
        self.assertIsNone(result.joints)

    def test_disconnected_controller_maps_to_unreachable(self) -> None:
        service = RobotArmIKService(_FakeArm(connected=False))
        result = service.query(_grasp())
        self.assertFalse(result.reachable)
        self.assertEqual(result.reason, "controller_not_connected")

    def test_non_base_frame_short_circuits(self) -> None:
        arm = _FakeArm()
        service = RobotArmIKService(arm)
        result = service.query(_grasp(frame=Frame.CAMERA))
        self.assertFalse(result.reachable)
        self.assertEqual(result.reason, "non_base_frame")
        # The arm must not be touched when the frame is wrong.
        self.assertEqual(arm.ik_calls, [])

    def test_seed_provider_overrides_arm_current_joints(self) -> None:
        arm = _FakeArm()
        custom = JointPositions([0.5] * 6)
        service = RobotArmIKService(arm, seed_provider=lambda: custom)
        service.query(_grasp())
        # Custom seed used, arm.get_joint_positions() should NOT be called.
        self.assertEqual(arm.seed_requests, 0)


class CachedIKServiceTests(unittest.TestCase):
    def test_near_duplicate_poses_hit_cache(self) -> None:
        arm = _FakeArm()
        cached = CachedIKService(RobotArmIKService(arm))
        cached.query(_grasp(0.0))
        cached.query(_grasp(0.1))  # sub-millimetre delta -> same bucket
        self.assertEqual(len(arm.ik_calls), 1)
        self.assertEqual(cached.hits, 1)
        self.assertEqual(cached.misses, 1)

    def test_distinct_poses_miss_cache(self) -> None:
        arm = _FakeArm()
        cached = CachedIKService(RobotArmIKService(arm))
        cached.query(_grasp(0.0))
        cached.query(_grasp(50.0))
        self.assertEqual(len(arm.ik_calls), 2)
        self.assertEqual(cached.hits, 0)
        self.assertEqual(cached.misses, 2)

    def test_eviction_respects_max_entries(self) -> None:
        arm = _FakeArm()
        cached = CachedIKService(RobotArmIKService(arm), max_entries=2)
        cached.query(_grasp(0.0))
        cached.query(_grasp(50.0))
        cached.query(_grasp(100.0))  # evicts the oldest (x=0)
        cached.query(_grasp(0.0))  # cache miss again -> arm called twice for x=0
        self.assertEqual(sum(1 for p in arm.ik_calls if abs(p.position_mm[0]) < 1e-6), 2)


class URAnalyticIKServiceTests(unittest.TestCase):
    def test_missing_optional_dep_raises_with_install_hint(self) -> None:
        try:
            import ur_ikfast  # type: ignore  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError) as ctx:
                URAnalyticIKService()
            self.assertIn("ur_ikfast", str(ctx.exception))
        else:  # pragma: no cover - exercised only on machines with ur_ikfast
            self.skipTest("ur_ikfast is installed; the import-error path cannot be tested here.")


if __name__ == "__main__":
    unittest.main()
