"""Tests for the JointLimitGuard."""

from __future__ import annotations

import math
import unittest


from src.config.schema.robot import (
    JointLimitSafetyConfig,
    RobotSafetyConfig,
    WorkspaceLimitsConfig,
)
from src.robot.core import (
    JointPositions,
    MotionCommand,
    RobotCapabilities,
)
from src.robot.safety import (
    JointLimitGuard,
    SafetyContext,
    SafetyPreflight,
    SafetyReason,
    UR_JOINT_LIMITS_DEG,
    resolve_joint_limits_deg,
)


class _StubArm:
    """Minimal arm stub exposing only ``capabilities``."""

    def __init__(self, vendor: str, model: str, dof: int = 6) -> None:
        self.capabilities = RobotCapabilities(
            vendor=vendor,
            model=model,
            dof=dof,
            supports_joint_move=True,
            supports_linear_move=True,
            supports_async_move=False,
            has_native_fk=False,
            has_native_ik=False,
            has_force_control=False,
            is_simulated=True,
        )


def _ctx(joints: JointPositions | None, *, arm: _StubArm | None = None) -> SafetyContext:
    return SafetyContext(
        command=MotionCommand.MOVE_TO,
        target_joints=joints,
        arm=arm,
    )


class ResolveJointLimitsTests(unittest.TestCase):
    def test_static_config_takes_priority(self) -> None:
        cfg = JointLimitSafetyConfig(
            min_deg=[-90.0] * 6, max_deg=[90.0] * 6,
        )
        lo, hi = resolve_joint_limits_deg(cfg, vendor="ur", model="ur5e")
        self.assertEqual(lo, [-90.0] * 6)
        self.assertEqual(hi, [90.0] * 6)

    def test_ur_vendor_table_used_when_static_absent(self) -> None:
        cfg = JointLimitSafetyConfig()
        lo, hi = resolve_joint_limits_deg(cfg, vendor="ur", model="ur5e")
        self.assertEqual(lo, list((-360.0,) * 6))
        self.assertEqual(hi, list((360.0,) * 6))

    def test_kuka_without_static_returns_none(self) -> None:
        cfg = JointLimitSafetyConfig()
        self.assertIsNone(
            resolve_joint_limits_deg(cfg, vendor="kuka", model="kr6-r900"),
        )

    def test_sim_without_static_returns_none(self) -> None:
        cfg = JointLimitSafetyConfig()
        self.assertIsNone(
            resolve_joint_limits_deg(cfg, vendor="sim", model="isaac-sim"),
        )

    def test_ur_table_covers_all_canonical_models(self) -> None:
        for name in ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20"):
            self.assertIn(name, UR_JOINT_LIMITS_DEG, msg=name)


class JointLimitGuardTests(unittest.TestCase):
    def _guard(self, **overrides) -> JointLimitGuard:
        cfg = JointLimitSafetyConfig(**overrides)
        return JointLimitGuard(cfg)

    def test_missing_target_joints_is_unavailable(self) -> None:
        guard = self._guard()
        d = guard.evaluate(_ctx(None, arm=_StubArm("ur", "ur5e")))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)

    def test_no_arm_and_no_static_is_unavailable(self) -> None:
        guard = self._guard()
        d = guard.evaluate(_ctx(JointPositions([0.0] * 6), arm=None))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)

    def test_kuka_without_static_limits_is_unavailable(self) -> None:
        guard = self._guard()
        d = guard.evaluate(_ctx(
            JointPositions([0.0] * 6), arm=_StubArm("kuka", "kr6-r900"),
        ))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)

    def test_ur_inside_envelope_accepts(self) -> None:
        guard = self._guard(margin_deg=5.0)
        d = guard.evaluate(_ctx(
            JointPositions([0.0] * 6), arm=_StubArm("ur", "ur5e"),
        ))
        self.assertTrue(d.accepted, msg=d.message)

    def test_ur_outside_envelope_rejects(self) -> None:
        guard = self._guard(margin_deg=5.0)
        # 359 deg + margin 5 => requires <= 355.  359 > 355 so it rejects.
        joints = JointPositions(
            [math.radians(359.0), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        d = guard.evaluate(_ctx(joints, arm=_StubArm("ur", "ur5e")))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.JOINT_LIMIT)
        self.assertEqual(d.detail["axis"], "0")

    def test_margin_subtracted_from_both_ends(self) -> None:
        guard = self._guard(margin_deg=10.0)
        joints = JointPositions(
            [math.radians(-355.5), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        d = guard.evaluate(_ctx(joints, arm=_StubArm("ur", "ur5e")))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.JOINT_LIMIT)

    def test_static_config_overrides_vendor_table(self) -> None:
        # Tighter operator-supplied envelope.
        guard = self._guard(
            min_deg=[-90.0] * 6, max_deg=[90.0] * 6, margin_deg=0.0,
        )
        # 120 deg is inside UR's ±360 but outside the operator's ±90.
        joints = JointPositions(
            [math.radians(120.0), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        d = guard.evaluate(_ctx(joints, arm=_StubArm("ur", "ur5e")))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.JOINT_LIMIT)

    def test_pathological_margin_returns_unavailable(self) -> None:
        # A non-pathological baseline (margin 45 within ±100) is fine; the
        # case that actually inverts the usable range is exercised below.
        guard2 = JointLimitGuard(
            JointLimitSafetyConfig(
                min_deg=[-30.0] * 6, max_deg=[30.0] * 6, margin_deg=45.0,
            )
        )
        joints = JointPositions([0.0] * 6)
        d = guard2.evaluate(_ctx(joints, arm=_StubArm("ur", "ur5e")))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)


class JointLimitGuardWiringTests(unittest.TestCase):
    """Verify ``from_safety_config`` wires the guard correctly."""

    def test_enforce_true_registers_guard(self) -> None:
        cfg = RobotSafetyConfig.model_validate({"joint_limits": {"enforce": True}})
        pf = SafetyPreflight.from_safety_config(cfg, WorkspaceLimitsConfig())
        self.assertIn("joint_limit", pf.guard_names)

    def test_enforce_false_omits_guard(self) -> None:
        cfg = RobotSafetyConfig.model_validate({"joint_limits": {"enforce": False}})
        pf = SafetyPreflight.from_safety_config(cfg, WorkspaceLimitsConfig())
        self.assertNotIn("joint_limit", pf.guard_names)


if __name__ == "__main__":
    unittest.main()
