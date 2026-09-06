"""Unit tests for the vendor-neutral singularity analyzer.

Uses a deterministic analytic 6-DoF mock arm so the numerical Jacobian (central differences over fk) is
platform-stable: joints 0-2 drive x/y/z position (100 mm/rad), joints 3-5 drive xyz-euler orientation.
With ``drop_joint5=True`` joint 5 has no effect -> the Jacobian is rank-deficient (a clean singularity).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_euler, to_euler
from src.robot.core import JointPositions, RobotSingularityRisk
from src.robot.safety.singularity import (
    SingularityGuard,
    SingularityThresholds,
    analyze_joint_singularity,
    assert_pose_not_singular,
)


class _MockArm:
    """Analytic arm with a controllable Jacobian (well-conditioned, or rank-deficient via drop_joint5)."""

    def __init__(self, drop_joint5: bool = False) -> None:
        self.drop_joint5 = drop_joint5

    def fk(self, joints: JointPositions) -> Pose:
        q = np.asarray(joints.values, dtype=np.float64)
        pos = np.array([q[0], q[1], q[2]], dtype=np.float64) * 100.0
        rz = 0.0 if self.drop_joint5 else float(q[5])
        quat = from_euler(np.array([q[3], q[4], rz]))
        return Pose(position_mm=pos, quaternion_xyzw=quat, frame=Frame.BASE, label="mock")

    def ik(self, pose: Pose, seed: JointPositions | None = None) -> JointPositions:
        pos = np.asarray(pose.position_mm, dtype=np.float64)
        eul = to_euler(np.asarray(pose.quaternion_xyzw, dtype=np.float64))
        return JointPositions([pos[0] / 100.0, pos[1] / 100.0, pos[2] / 100.0, eul[0], eul[1], eul[2]])


# Small angles, away from gimbal lock (pitch != +/-90 deg).
_WELL = JointPositions([1.0, 0.5, 2.0, 0.2, 0.3, -0.1])


class TestAnalyzeJointSingularity:
    def test_well_conditioned_not_near_singular(self) -> None:
        rep = analyze_joint_singularity(_MockArm(), _WELL)
        assert rep.rank == 6
        assert rep.expected_rank == 6
        assert not rep.is_near_singularity
        assert rep.reasons == ()
        assert rep.condition_number < SingularityThresholds().max_condition_number
        assert rep.min_singular_value > SingularityThresholds().min_singular_value

    def test_rank_deficient_is_near_singular(self) -> None:
        rep = analyze_joint_singularity(_MockArm(drop_joint5=True), _WELL)
        assert rep.rank == 5
        assert rep.is_near_singularity
        assert any("rank_deficient" in r for r in rep.reasons)
        assert any("min_sigma_below_threshold" in r for r in rep.reasons)

    def test_custom_threshold_flags_condition(self) -> None:
        # A deliberately strict condition ceiling flags the otherwise-fine arm.
        rep = analyze_joint_singularity(
            _MockArm(), _WELL, thresholds=SingularityThresholds(max_condition_number=1.0)
        )
        assert rep.is_near_singularity
        assert any("condition_number_above_threshold" in r for r in rep.reasons)

    def test_delta_rad_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            analyze_joint_singularity(_MockArm(), _WELL, delta_rad=0.0)


class TestAssertPoseNotSingular:
    def test_returns_joints_when_safe(self) -> None:
        arm = _MockArm()
        out = assert_pose_not_singular(arm, arm.fk(_WELL))
        assert isinstance(out, JointPositions)
        assert out.dof == 6

    def test_raises_on_singular(self) -> None:
        arm = _MockArm(drop_joint5=True)
        with pytest.raises(RobotSingularityRisk):
            assert_pose_not_singular(arm, arm.fk(_WELL))


class TestSingularityGuard:
    def test_analyze_and_require_safe(self) -> None:
        arm = _MockArm()
        guard = SingularityGuard(arm)
        pose = arm.fk(_WELL)
        assert not guard.analyze_pose(pose).is_near_singularity
        assert isinstance(guard.require_safe_pose(pose), JointPositions)

    def test_require_safe_raises_on_singular(self) -> None:
        arm = _MockArm(drop_joint5=True)
        guard = SingularityGuard(arm)
        with pytest.raises(RobotSingularityRisk):
            guard.require_safe_pose(arm.fk(_WELL))
