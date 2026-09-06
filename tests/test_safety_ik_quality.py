"""Tests for the IKQualityGuard."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.config.schema.robot import (
    IkQualitySafetyConfig,
    JointLimitSafetyConfig,
    RobotSafetyConfig,
    WorkspaceLimitsConfig,
)
from src.geometry import Frame, Pose
from src.geometry.quaternion import IDENTITY_QUAT_XYZW
from src.robot.core import (
    JointPositions,
    MotionCommand,
    RobotCapabilities,
)
from src.robot.safety import (
    IKQualityGuard,
    SafetyContext,
    SafetyPreflight,
    SafetyReason,
)


class _IdentityFkArm:
    """Arm stub with a trivial linear FK (one axis = position along X).

    Lets us drive a real numerical Jacobian through
    :func:`analyze_joint_singularity` without depending on a real
    kinematic chain.
    """

    def __init__(self, vendor: str = "sim", model: str = "stub", dof: int = 6) -> None:
        self.capabilities = RobotCapabilities(
            vendor=vendor, model=model, dof=dof,
            supports_joint_move=True, supports_linear_move=True,
            supports_async_move=False,
            has_native_fk=True, has_native_ik=False, has_force_control=False,
            is_simulated=True,
        )

    def fk(self, joints: JointPositions) -> Pose:
        # Each axis contributes a tiny linear translation. Distinct
        # axes -> well-conditioned Jacobian.
        x = float(np.sum(joints.values * np.array([100.0, 80.0, 60.0, 40.0, 30.0, 25.0])))
        y = float(np.sum(joints.values * np.array([0.0, 50.0, 0.0, 30.0, 0.0, 20.0])))
        z = float(np.sum(joints.values * np.array([0.0, 0.0, 50.0, 0.0, 30.0, 0.0])))
        return Pose(
            position_mm=np.array([x, y, z], dtype=np.float64),
            quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
            frame=Frame.BASE,
            label="stub-fk",
        )


def _ctx(
    target_joints: JointPositions | None,
    *,
    current_joints: JointPositions | None = None,
    arm: _IdentityFkArm | None = None,
) -> SafetyContext:
    return SafetyContext(
        command=MotionCommand.MOVE_TO,
        target_joints=target_joints,
        current_joints=current_joints,
        arm=arm,
    )


class IKQualityGuardTests(unittest.TestCase):
    def _guard(self, **overrides) -> IKQualityGuard:
        cfg = IkQualitySafetyConfig(**overrides)
        # Use ±360 deg envelope so the limit-proximity check doesn't
        # trigger by accident on unrelated tests.
        jl = JointLimitSafetyConfig(
            min_deg=[-360.0] * 6, max_deg=[360.0] * 6,
        )
        return IKQualityGuard(cfg, joint_limits_config=jl)

    def test_missing_target_joints_is_unavailable(self) -> None:
        guard = self._guard()
        d = guard.evaluate(_ctx(None))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)

    def test_wrong_dof_rejects(self) -> None:
        guard = self._guard()
        arm = _IdentityFkArm(dof=6)
        joints = JointPositions([0.0] * 7)  # 7-DoF answer for a 6-DoF arm
        d = guard.evaluate(_ctx(joints, arm=arm))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.IK_QUALITY)
        self.assertEqual(d.detail["reason"], "wrong_dof")

    def test_joint_jump_rejects(self) -> None:
        guard = self._guard(max_jump_rad=0.5)
        current = JointPositions([0.0] * 6)
        target = JointPositions([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # 1 rad jump
        d = guard.evaluate(_ctx(target, current_joints=current))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.IK_QUALITY)
        self.assertEqual(d.detail["reason"], "joint_jump")
        self.assertEqual(d.detail["axis"], "2")

    def test_small_jump_accepts(self) -> None:
        guard = self._guard(max_jump_rad=0.5)
        current = JointPositions([0.0] * 6)
        target = JointPositions([0.1] * 6)
        d = guard.evaluate(_ctx(target, current_joints=current))
        self.assertTrue(d.accepted, msg=d.message)

    def test_limit_proximity_rejects(self) -> None:
        # Custom joint-limit config: tight ±10 deg envelope.
        cfg = IkQualitySafetyConfig(limit_proximity_deg=2.0)
        jl = JointLimitSafetyConfig(
            min_deg=[-10.0] * 6, max_deg=[10.0] * 6,
        )
        guard = IKQualityGuard(cfg, joint_limits_config=jl)
        arm = _IdentityFkArm()
        # 9 deg is only 1 deg from +10 limit; buffer is 2 deg.
        joints = JointPositions([math.radians(9.0), 0.0, 0.0, 0.0, 0.0, 0.0])
        d = guard.evaluate(_ctx(joints, arm=arm))
        self.assertFalse(d.accepted)
        self.assertEqual(d.detail["reason"], "limit_proximity")

    def test_jacobian_runs_when_arm_has_native_fk(self) -> None:
        # When the test stub advertises has_native_fk=True the guard
        # tries to probe the Jacobian. The stub's purely linear-in-
        # position FK is rank-3 so any thresholded singularity check
        # would fail; the point of this test is only that no exception
        # escapes the guard and a deterministic decision is returned.
        guard = self._guard()
        arm = _IdentityFkArm()
        joints = JointPositions([0.0] * 6)
        d = guard.evaluate(_ctx(joints, arm=arm))
        # Either accepted or rejected with IK_QUALITY is acceptable here;
        # the contract under test is "the probe ran without raising".
        self.assertIn(
            d.reason,
            (None, SafetyReason.IK_QUALITY),
            msg=d.message,
        )

    def test_skips_singularity_when_no_native_fk(self) -> None:
        # When the arm advertises has_native_fk=False the singularity
        # probe is skipped; the rest of the guard still runs.
        guard = self._guard()
        arm = _IdentityFkArm()
        # Patch the capability flag.
        from dataclasses import replace
        arm.capabilities = replace(arm.capabilities, has_native_fk=False)
        joints = JointPositions([0.0] * 6)
        d = guard.evaluate(_ctx(joints, arm=arm))
        self.assertTrue(d.accepted, msg=d.message)

    def test_fk_failure_returns_unavailable(self) -> None:
        class _BadFkArm(_IdentityFkArm):
            def fk(self, joints):  # noqa: ARG002
                raise RuntimeError("controller dropped")

        guard = self._guard()
        d = guard.evaluate(_ctx(JointPositions([0.0] * 6), arm=_BadFkArm()))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)

    def test_fk_failure_returns_unavailable_for_the_errors_a_REAL_arm_raises(self) -> None:
        """The guard must fail CLOSED for the exception types real hardware actually produces.

        MEASURED 2026-08-09: the handler caught ``(RuntimeError, OSError, ValueError)``. In
        ``core/errors.py``, ``IsaacNotAvailableError`` subclasses ``RobotError`` AND ``RuntimeError``,
        so the sim always took the typed-UNAVAILABLE branch and this guard looked correct. But
        ``RobotKinematicsError`` and ``RobotConnectionError`` subclass ``RobotError`` ONLY -- and those
        are exactly what ``URRobotArm.fk()`` raises (drivers/ur/arm.py:463-471) when the connection is
        closed or the controller refuses the solve. With ``ik_quality.enforce: true`` shipped
        (robot.yaml:111) and the probe running on every gated move, a real cell would take an unhandled
        traceback mid-pick instead of a typed safety verdict. The test above passes either way, which
        is precisely why it did not catch this: it raises the one type that is NOT the real one.
        """
        from src.robot.core.errors import (
            RobotConnectionError,
            RobotKinematicsError,
        )

        for exc_type in (RobotKinematicsError, RobotConnectionError):
            with self.subTest(exception=exc_type.__name__):
                self.assertFalse(
                    issubclass(exc_type, (RuntimeError, OSError, ValueError)),
                    f"{exc_type.__name__} is a plain RuntimeError now -- this test no longer proves "
                    f"anything; check core/errors.py before relaxing it.",
                )

                class _BadFkArm(_IdentityFkArm):
                    def fk(self, joints):  # noqa: ARG002
                        raise exc_type("controller dropped mid-pick")

                d = self._guard().evaluate(_ctx(JointPositions([0.0] * 6), arm=_BadFkArm()))
                self.assertFalse(d.accepted)
                self.assertIs(d.reason, SafetyReason.UNAVAILABLE)
                self.assertEqual(d.detail.get("exception"), exc_type.__name__)


class IKQualityWiringTests(unittest.TestCase):
    def test_enforce_true_registers_guard(self) -> None:
        cfg = RobotSafetyConfig.model_validate({"ik_quality": {"enforce": True}})
        pf = SafetyPreflight.from_safety_config(cfg, WorkspaceLimitsConfig())
        self.assertIn("ik_quality", pf.guard_names)

    def test_enforce_false_omits_guard(self) -> None:
        cfg = RobotSafetyConfig.model_validate({"ik_quality": {"enforce": False}})
        pf = SafetyPreflight.from_safety_config(cfg, WorkspaceLimitsConfig())
        self.assertNotIn("ik_quality", pf.guard_names)


if __name__ == "__main__":
    unittest.main()
