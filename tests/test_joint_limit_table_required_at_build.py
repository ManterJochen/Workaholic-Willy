"""A cell whose joint-limit guard has no table to enforce must be refused where the key can be filled in.

**A FAIL-CLOSED CELL AND A DEAD CELL LOOK THE SAME FROM THE OUTSIDE, AND ONE OF THEM WAS BOTH.**
Measured 2026-09-10 on the shipped ``web`` profile (the only ``vendor: kuka`` config in this tree):
``resolve_joint_limits_deg`` returned ``None``, so ``JointLimitGuard`` answered ``UNAVAILABLE`` for an
all-zeros joint target, and ``SafetyPreflight`` turned that into ``controller_rejected``. Every motion
was refused, legal or absurd, and the refusal named the CONTROLLER while the fault was a missing
config key. The operator reads "controller_rejected" and goes to look at the KRC.

The refusal was safe and it was unusable, and it surfaced one move at a time in the one place nobody
can fix it. The build-time check below moves it to boot, where the message names the fix: an arm's
envelope is its description's since 2026-10-17 (``config/arms/<model>.yaml``; no tree writes a table).

**THE TESTS BELOW NAME NO VENDOR.** A test that said "kuka" would stop meaning anything the day a
second vendor landed the same way, which is exactly how this one arrived: only Universal Robots ship a
built-in table. The property is "an arm that runs the joint-limit guard must be able to resolve a
table for its own capabilities", and it is asked of a synthetic vendor's arm, registered in the dummy's
slot for the length of each test.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.robot.core import RobotCapabilities, RobotVendor
from src.robot.drivers import create_arm, register_arm_driver, registry
from src.robot.drivers.dummy import DummyRobotArm
from src.robot.safety import JointLimitGuard, SafetyPreflight
from src.robot.safety.joint_limits import JointLimitTableMissing

#: The slot the tests below register in, for the length of each test: the dummy's own factory is put back after
#: it. Every member has a driver since the reserved names left (2026-10-17), so "the next vendor to arrive" borrows one.
_SLOT = RobotVendor.DUMMY
#: What that arm says it is: a vendor no table and no enum member describes.
_FUTURE = "acme"


class _FutureVendorArm(DummyRobotArm):
    """A driver for a vendor with no built-in limit table, wired the way every real driver is.

    Inherits the dummy's motion surface (none of it is exercised here) and overrides the two things
    the build-time check reads: what the arm says it IS, and what pipeline it says it runs.
    """

    def __init__(self, *, pipeline: SafetyPreflight | None = None) -> None:
        super().__init__()
        self._caps = RobotCapabilities(
            vendor=_FUTURE, model="fr-42", dof=6,
            has_native_fk=False,
            is_simulated=False,
                                     controller=None, owns_a_controller=True,
        )
        if pipeline is None:
            pipeline = SafetyPreflight([JointLimitGuard(half_turn_about_home=False)])
        self._pipeline = pipeline

    @property
    def capabilities(self) -> RobotCapabilities:
        return self._caps

    @property
    def safety_preflight(self) -> SafetyPreflight | None:
        return self._pipeline


class _UngatedFutureVendorArm(DummyRobotArm):
    """Same new vendor, no guard pipeline at all: UNGATED is a stated decision, not this defect."""

    @property
    def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(
            vendor=_FUTURE, model="fr-42", dof=6,
            has_native_fk=False,
            is_simulated=False,
                                     controller=None, owns_a_controller=True,
        )


class ANewVendorWithNoLimitTableIsRefusedAtBuild(unittest.TestCase):
    """The chokepoint is ``create_arm``, because that is the one line every cell boot passes."""

    def _register(self, factory) -> None:
        dummy = registry._REGISTRY[_SLOT]
        register_arm_driver(_SLOT, overwrite=True)(factory)
        self.addCleanup(register_arm_driver(_SLOT, overwrite=True), dummy)

    def test_no_table_anywhere_refuses_the_build(self) -> None:
        self._register(lambda **_: _FutureVendorArm())
        with self.assertRaises(JointLimitTableMissing) as caught:
            create_arm(_SLOT)
        message = str(caught.exception)
        self.assertIn("config/arms/<model>.yaml", message, f"the refusal must name what fixes it: {message}")
        self.assertIn("limit_deg", message, f"the refusal must name what fixes it: {message}")
        self.assertIn(
            _FUTURE, message,
            f"the refusal must name the vendor whose table is missing: {message}",
        )

    def test_a_described_arm_builds_the_same_arm(self) -> None:
        """The control. Without it the test above would also pass for a check that refuses everything: the same arm,
        its model described (the registry stood in for, as the description a vendor's port would add)."""
        self._register(lambda **_: _FutureVendorArm())
        with mock.patch("src.robot.safety.joint_limits.resolve_joint_limits_deg",
                        return_value=([-180.0] * 6, [180.0] * 6)):
            arm = create_arm(_SLOT)
        self.assertEqual(arm.capabilities.vendor, _FUTURE)

    def test_an_arm_that_states_it_gates_nothing_still_builds(self) -> None:
        """UNGATED is a decision the operator made; this check refuses a broken guard, not no guard."""
        self._register(lambda **_: _UngatedFutureVendorArm())
        arm = create_arm(_SLOT)
        self.assertIsNone(arm.safety_preflight)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
