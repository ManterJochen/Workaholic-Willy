"""Tests for the MotionContinuityGuard, at the arm's guard numbers (``GuardNumbers``)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.config.schema.robot import (
    RobotSafetyConfig,
    WorkspaceLimitsConfig,
)
from src.geometry import Frame, Pose
from src.geometry.quaternion import IDENTITY_QUAT_XYZW, from_axis_angle
from src.robot.core import JointPositions, MotionCommand
from src.robot.safety import (
    MotionContinuityGuard,
    SafetyContext,
    SafetyPreflight,
    SafetyReason,
)
from src.robot.safety.guard_numbers import CELL, GuardNumbers


def _pose(x_mm: float, *, quat=None, label: str = "p") -> Pose:
    return Pose(
        position_mm=np.array([x_mm, 0.0, 0.0], dtype=np.float64),
        quaternion_xyzw=(quat if quat is not None else IDENTITY_QUAT_XYZW.copy()),
        frame=Frame.BASE,
        label=label,
    )


class MotionContinuityGuardTests(unittest.TestCase):
    def _guard(self, **overrides) -> MotionContinuityGuard:
        return MotionContinuityGuard(GuardNumbers(**overrides))

    def test_every_real_arm_runs_the_cell_s_numbers(self) -> None:
        self.assertEqual((45.0, 30.0, 250.0),
                         (CELL.max_joint_step_deg, CELL.max_orientation_step_deg, CELL.max_tcp_step_mm))

    def test_no_previous_target_accepts(self) -> None:
        guard = self._guard()
        ctx = SafetyContext(
            command=MotionCommand.MOVE_TO,
            target_pose=_pose(0.0),
            target_joints=JointPositions([0.0] * 6),
        )
        self.assertTrue(guard.evaluate(ctx).accepted)

    def test_large_joint_step_rejects(self) -> None:
        guard = self._guard(max_joint_step_deg=10.0)
        prev = JointPositions([0.0] * 6)
        now = JointPositions(
            [math.radians(45.0), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        ctx = SafetyContext(
            command=MotionCommand.MOVE_TO,
            target_joints=now,
            last_target_joints=prev,
        )
        d = guard.evaluate(ctx)
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.CONTINUITY)
        self.assertEqual(d.detail["reason"], "joint_step")

    def test_small_joint_step_accepts(self) -> None:
        guard = self._guard(max_joint_step_deg=10.0)
        prev = JointPositions([0.0] * 6)
        now = JointPositions(
            [math.radians(5.0), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        ctx = SafetyContext(
            command=MotionCommand.MOVE_TO,
            target_joints=now,
            last_target_joints=prev,
        )
        self.assertTrue(guard.evaluate(ctx).accepted)

    def test_large_tcp_step_rejects(self) -> None:
        guard = self._guard(max_tcp_step_mm=50.0)
        ctx = SafetyContext(
            command=MotionCommand.MOVE_TO,
            target_pose=_pose(200.0),
            last_target_pose=_pose(0.0),
        )
        d = guard.evaluate(ctx)
        self.assertFalse(d.accepted)
        self.assertEqual(d.detail["reason"], "tcp_step")

    def test_large_orientation_step_rejects(self) -> None:
        guard = self._guard(max_orientation_step_deg=10.0)
        # 90 deg about Z, identity quat -> obvious orientation jump.
        rot_z_90 = from_axis_angle(
            np.array([0.0, 0.0, math.radians(90.0)], dtype=np.float64)
        )
        ctx = SafetyContext(
            command=MotionCommand.MOVE_TO,
            target_pose=_pose(0.0, quat=rot_z_90),
            last_target_pose=_pose(0.0),
        )
        d = guard.evaluate(ctx)
        self.assertFalse(d.accepted)
        self.assertEqual(d.detail["reason"], "orientation_step")

    def test_dof_mismatch_skips_joint_check(self) -> None:
        guard = self._guard(max_joint_step_deg=1.0)
        prev = JointPositions([0.0] * 6)
        now = JointPositions([1.0] * 7)
        ctx = SafetyContext(
            command=MotionCommand.MOVE_TO,
            target_joints=now,
            last_target_joints=prev,
        )
        # No pose data; nothing to reject.
        self.assertTrue(guard.evaluate(ctx).accepted)


class MotionContinuityWiringTests(unittest.TestCase):
    def test_every_cell_wires_the_guard(self) -> None:
        pf = SafetyPreflight.from_safety_config(RobotSafetyConfig(), WorkspaceLimitsConfig(), arm_model=None)
        self.assertIn("motion_continuity", pf.guard_names)

    def test_the_switch_that_left_it_out_is_refused(self) -> None:
        """No cell switches a guard off since 2026-10-17 (the owner's cleanup)."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            RobotSafetyConfig.model_validate({"motion_continuity": {"enforce": False}})


if __name__ == "__main__":
    unittest.main()
