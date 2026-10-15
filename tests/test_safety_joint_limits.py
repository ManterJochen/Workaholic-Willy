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
    resolve_joint_limits_deg,
)


class _StubArm:
    """Minimal arm stub exposing only ``capabilities``."""

    def __init__(self, vendor: str, model: str, dof: int = 6) -> None:
        self.capabilities = RobotCapabilities(
            vendor=vendor,
            model=model,
            dof=dof,
            has_native_fk=False,
            is_simulated=True,
            controller=None, owns_a_controller=vendor not in ("sim", "dummy"),
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
        lo, hi = resolve_joint_limits_deg(cfg, model="ur5e")
        self.assertEqual(lo, [-90.0] * 6)
        self.assertEqual(hi, [90.0] * 6)

    def test_a_described_arm_gets_its_envelope_when_static_absent(self) -> None:
        cfg = JointLimitSafetyConfig()
        lo, hi = resolve_joint_limits_deg(cfg, model="ur5e")
        self.assertEqual(lo, list((-360.0,) * 6))
        self.assertEqual(hi, list((360.0,) * 6))

    def test_an_arm_nobody_describes_gets_none_without_static(self) -> None:
        """A KUKA's model, and the ``isaac-sim`` the Isaac arm reported until P0d D5: no description, no envelope."""
        cfg = JointLimitSafetyConfig()
        for model in ("kr6-r900", "isaac-sim", "dummy-6dof"):
            with self.subTest(model=model):
                self.assertIsNone(resolve_joint_limits_deg(cfg, model=model))

    def test_the_vendor_does_not_decide_any_more(self) -> None:
        """The case that pinned the vendor gate, inverted (the owner, in aurora, 2026-09-22).

        The gate was kept "until PY-b" on the premise that the sim's joint-limit wrap had never run. It had: the sim
        profile's static lists answer before the gate, and the wrap measurably turned a wrist on the dense cell on
        2026-09-17. What the gate decided was a vendor="sim" tree with no lists, which it left UNAVAILABLE on every
        motion. Such a tree's arm reports the UR it drives, and a guard holding it now enforces that UR's envelope.
        The resolver takes no vendor at all, so none can creep back in.
        """
        cfg = JointLimitSafetyConfig()
        self.assertEqual(([-360.0] * 6, [360.0] * 6), resolve_joint_limits_deg(cfg, model="ur5e"))
        with self.assertRaises(TypeError):
            resolve_joint_limits_deg(cfg, vendor="sim", model="ur5e")  # type: ignore[call-arg]
        inside = JointLimitGuard(cfg).evaluate(_ctx(JointPositions([0.0] * 6), arm=_StubArm("sim", "ur5e")))
        self.assertTrue(inside.accepted, inside.message)
        past = JointLimitGuard(cfg).evaluate(_ctx(
            JointPositions([math.radians(361.0), 0.0, 0.0, 0.0, 0.0, 0.0]), arm=_StubArm("sim", "ur5e")))
        self.assertIs(past.reason, SafetyReason.JOINT_LIMIT, "the envelope is enforced, not only found")

    def test_the_sim_profile_s_own_lists_still_win(self) -> None:
        """The shipped sim cell is unchanged by the gate going: its declared lists answer before any description."""
        from src.willy_sim.config import load_sim_config, require_robot

        limits = require_robot(load_sim_config()).safety.joint_limits
        self.assertIsNotNone(limits.min_deg, "the control: the sim profile declares its lists")
        self.assertEqual((list(limits.min_deg or ()), list(limits.max_deg or ())),
                         resolve_joint_limits_deg(limits, model="ur5e"))

    def test_the_envelope_comes_from_the_registry_now(self) -> None:
        """P0d D3c. Written while the hand-kept table stood beside the registry and agreed with it, so no model told
        them apart; a changed description does. With the registry answering a narrower elbow, the resolver says so.
        """
        from unittest import mock

        from src.config.arms import load_arm
        from src.robot.safety import joint_limits

        real = load_arm("ur5e")
        joints = list(real.joints)
        joints[2] = joints[2].model_copy(update={"limit_deg": (-90.0, 90.0)})
        narrowed = real.model_copy(update={"joints": tuple(joints)})
        joint_limits._described_limits_deg.cache_clear()
        try:
            with mock.patch("src.config.arms.load_arm", return_value=narrowed):
                lo, hi = resolve_joint_limits_deg(JointLimitSafetyConfig(), model="ur5e")
        finally:
            joint_limits._described_limits_deg.cache_clear()
        self.assertEqual((-90.0, 90.0), (lo[2], hi[2]))

    def test_every_described_arm_gets_its_own_envelope(self) -> None:
        """The resolver the guard calls answers each description's `limit_deg`, joint for joint."""
        from src.config.arms import available_arms, load_arm

        for name in available_arms():
            joints = load_arm(name).joints
            with self.subTest(model=name):
                self.assertEqual(([j.limit_deg[0] for j in joints], [j.limit_deg[1] for j in joints]),
                                 resolve_joint_limits_deg(JointLimitSafetyConfig(), model=name))

    def test_every_canonical_model_has_an_envelope(self) -> None:
        for name in ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20"):
            with self.subTest(model=name):
                self.assertIsNotNone(resolve_joint_limits_deg(JointLimitSafetyConfig(), model=name))


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
        pf = SafetyPreflight.from_safety_config(cfg, WorkspaceLimitsConfig(), arm_model=None)
        self.assertIn("joint_limit", pf.guard_names)

    def test_enforce_false_omits_guard(self) -> None:
        cfg = RobotSafetyConfig.model_validate({"joint_limits": {"enforce": False}})
        pf = SafetyPreflight.from_safety_config(cfg, WorkspaceLimitsConfig(), arm_model=None)
        self.assertNotIn("joint_limit", pf.guard_names)


if __name__ == "__main__":
    unittest.main()
