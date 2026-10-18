"""Tests for the JointLimitGuard: the arm's own envelope, 5 degrees inside each end, half a turn about home.

No tree writes a joint table, a margin or the window since 2026-10-17 (the owner's cleanup): every arm's envelope is
its description's, the margin every cell's 5 degrees (``JOINT_LIMIT_MARGIN_DEG``), and every real arm keeps every
full-turn joint within half a turn of home; the Isaac arm states it keeps its whole envelope
(``GuardNumbers.half_turn_about_home``).
"""

from __future__ import annotations

import math
import unittest


from src.config.schema.robot import (
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
from src.robot.safety.joint_limits import JOINT_LIMIT_MARGIN_DEG


class _StubArm:
    """Minimal arm stub exposing ``capabilities`` and, where given, the home it states."""

    def __init__(self, vendor: str, model: str, dof: int = 6, home: "list[float] | None" = None) -> None:
        self.capabilities = RobotCapabilities(
            vendor=vendor,
            model=model,
            dof=dof,
            has_native_fk=False,
            is_simulated=True,
            controller=None, owns_a_controller=vendor not in ("sim", "dummy"),
        )
        if home is not None:
            self.home_joint_positions = tuple(home)


def _ctx(joints: JointPositions | None, *, arm: _StubArm | None = None) -> SafetyContext:
    return SafetyContext(
        command=MotionCommand.MOVE_TO,
        target_joints=joints,
        arm=arm,
    )


#: The guard over the whole envelope, as the Isaac arm runs it: what these read is the envelope and the margin.
def _whole() -> JointLimitGuard:
    return JointLimitGuard(half_turn_about_home=False)


class ResolveJointLimitsTests(unittest.TestCase):
    def test_a_described_arm_gets_its_envelope(self) -> None:
        lo, hi = resolve_joint_limits_deg(model="ur5e")  # type: ignore[misc]
        self.assertEqual(lo, list((-360.0,) * 6))
        self.assertEqual(hi, list((360.0,) * 6))

    def test_an_arm_nobody_describes_gets_none(self) -> None:
        """A KUKA's model, and the ``isaac-sim`` the Isaac arm reported until P0d D5: no description, no envelope."""
        for model in ("kr6-r900", "isaac-sim", "dummy-6dof"):
            with self.subTest(model=model):
                self.assertIsNone(resolve_joint_limits_deg(model=model))

    def test_the_vendor_does_not_decide_any_more(self) -> None:
        """The case that pinned the vendor gate, inverted (the owner, in aurora, 2026-09-22).

        What the gate decided was a vendor="sim" tree, which it left UNAVAILABLE on every motion. Such a tree's arm
        reports the UR it drives, and a guard holding it enforces that UR's envelope, the +-360 the sim profile wrote
        out until 2026-10-17. The resolver takes no vendor and no table at all, so neither can creep back in.
        """
        self.assertEqual(([-360.0] * 6, [360.0] * 6), resolve_joint_limits_deg(model="ur5e"))
        with self.assertRaises(TypeError):
            resolve_joint_limits_deg(vendor="sim", model="ur5e")  # type: ignore[call-arg]
        inside = _whole().evaluate(_ctx(JointPositions([0.0] * 6), arm=_StubArm("sim", "ur5e")))
        self.assertTrue(inside.accepted, inside.message)
        past = _whole().evaluate(_ctx(
            JointPositions([math.radians(361.0), 0.0, 0.0, 0.0, 0.0, 0.0]), arm=_StubArm("sim", "ur5e")))
        self.assertIs(past.reason, SafetyReason.JOINT_LIMIT, "the envelope is enforced, not only found")

    def test_the_old_keys_are_refused(self) -> None:
        from pydantic import ValidationError

        for block in ({"min_deg": [-360.0] * 6, "max_deg": [360.0] * 6}, {"margin_deg": 5.0},
                      {"within_half_turn_of_home": True}):
            with self.subTest(block=block), self.assertRaises(ValidationError):
                RobotSafetyConfig.model_validate({"joint_limits": block})

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
                lo, hi = resolve_joint_limits_deg(model="ur5e")  # type: ignore[misc]
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
                                 resolve_joint_limits_deg(model=name))

    def test_every_canonical_model_has_an_envelope(self) -> None:
        for name in ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20"):
            with self.subTest(model=name):
                self.assertIsNotNone(resolve_joint_limits_deg(model=name))


class JointLimitGuardTests(unittest.TestCase):
    def test_the_margin_is_every_cell_s_5_degrees(self) -> None:
        self.assertEqual(5.0, JOINT_LIMIT_MARGIN_DEG)
        self.assertEqual(5.0, JointLimitGuard().margin_deg)

    def test_missing_target_joints_is_unavailable(self) -> None:
        d = _whole().evaluate(_ctx(None, arm=_StubArm("ur", "ur5e")))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)

    def test_no_arm_is_unavailable(self) -> None:
        d = _whole().evaluate(_ctx(JointPositions([0.0] * 6), arm=None))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)

    def test_an_arm_nobody_describes_is_unavailable(self) -> None:
        d = _whole().evaluate(_ctx(
            JointPositions([0.0] * 6), arm=_StubArm("kuka", "kr6-r900"),
        ))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)

    def test_ur_inside_envelope_accepts(self) -> None:
        d = _whole().evaluate(_ctx(
            JointPositions([0.0] * 6), arm=_StubArm("ur", "ur5e"),
        ))
        self.assertTrue(d.accepted, msg=d.message)

    def test_ur_outside_envelope_rejects(self) -> None:
        # 359 deg + margin 5 => requires <= 355.  359 > 355 so it rejects.
        joints = JointPositions(
            [math.radians(359.0), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        d = _whole().evaluate(_ctx(joints, arm=_StubArm("ur", "ur5e")))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.JOINT_LIMIT)
        self.assertEqual(d.detail["axis"], "0")

    def test_margin_subtracted_from_both_ends(self) -> None:
        joints = JointPositions(
            [math.radians(-355.5), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        d = _whole().evaluate(_ctx(joints, arm=_StubArm("ur", "ur5e")))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.JOINT_LIMIT)

    def test_every_real_arm_keeps_half_a_turn_about_home(self) -> None:
        """The default, every real arm's: the base at 190 deg from a home at 0 is inside the +-360 envelope and half
        a turn and more from home, so it is refused; at 170 deg it is not."""
        home = [0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0]
        arm = _StubArm("ur", "ur5e", home=home)
        guard = JointLimitGuard()
        self.assertTrue(guard.within_half_turn_of_home)
        far = guard.evaluate(_ctx(JointPositions([math.radians(190.0), *home[1:]]), arm=arm))
        self.assertIs(far.reason, SafetyReason.JOINT_LIMIT, far.message)
        self.assertIn("half a turn either side of home", far.message or "")
        near = guard.evaluate(_ctx(JointPositions([math.radians(170.0), *home[1:]]), arm=arm))
        self.assertTrue(near.accepted, near.message)
        self.assertTrue(_whole().evaluate(_ctx(JointPositions([math.radians(190.0), *home[1:]]), arm=arm)).accepted,
                        "the control: the whole envelope holds it")

    def test_a_home_that_leaves_an_axis_no_range_is_unavailable(self) -> None:
        """A home written in degrees reads as radians: 540 rad-written-as-degrees on the base puts the window outside
        the table, and the margin eats what is left."""
        home = [math.radians(540.0), -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0]
        d = JointLimitGuard().evaluate(_ctx(JointPositions([0.0] * 6), arm=_StubArm("ur", "ur5e", home=home)))
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)


class JointLimitGuardWiringTests(unittest.TestCase):
    """Verify ``from_safety_config`` wires the guard correctly."""

    def test_every_cell_wires_the_guard(self) -> None:
        pf = SafetyPreflight.from_safety_config(RobotSafetyConfig(), WorkspaceLimitsConfig(), arm_model=None)
        self.assertIn("joint_limit", pf.guard_names)

    def test_the_switch_that_left_it_out_is_refused(self) -> None:
        """No cell switches a guard off since 2026-10-17 (the owner's cleanup)."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            RobotSafetyConfig.model_validate({"joint_limits": {"enforce": False}})


if __name__ == "__main__":
    unittest.main()
