"""A cell whose joint-limit guard has no table to enforce must be refused where the key can be filled in.

**A FAIL-CLOSED CELL AND A DEAD CELL LOOK THE SAME FROM THE OUTSIDE, AND ONE OF THEM WAS BOTH.**
Measured 2026-09-10 on the shipped ``web`` profile (the only ``vendor: kuka`` config in this tree):
``resolve_joint_limits_deg`` returned ``None``, so ``JointLimitGuard`` answered ``UNAVAILABLE`` for an
all-zeros joint target, and ``SafetyPreflight`` turned that into ``controller_rejected``. Every motion
was refused, legal or absurd, and the refusal named the CONTROLLER while the fault was a missing
config key. The operator reads "controller_rejected" and goes to look at the KRC.

The refusal was safe and it was unusable, and it surfaced one move at a time in the one place nobody
can fix it. The build-time check below moves it to boot, where the message can name
``robot.safety.joint_limits.min_deg`` / ``max_deg``.

**THE TESTS BELOW NAME NO VENDOR.** A test that said "kuka" would stop meaning anything the day a
second vendor landed the same way, which is exactly how this one arrived: only Universal Robots ship a
built-in table, and ``drivers/franka/`` and ``drivers/ros2/`` are declared empty slots waiting to be
filled. The property is "an arm that enforces the joint-limit guard must be able to resolve a table
for its own capabilities", and it is asked of a synthetic vendor registered for the length of the test.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from src.config.schema.robot import (
    JointLimitSafetyConfig,
    RobotSafetyConfig,
    WorkspaceLimitsConfig,
)
from src.robot.core import RobotCapabilities, RobotVendor
from src.robot.drivers import (
    create_arm,
    register_arm_driver,
    unregister_arm_driver,
)
from src.robot.drivers.dummy import DummyRobotArm
from src.robot.safety import JointLimitGuard, SafetyPreflight
from src.robot.safety import joint_limits as joint_limits_module
from src.robot.safety.joint_limits import JointLimitTableMissing

#: The vendor the tests below register. A reserved slot with no driver of its own, so registering a
#: factory here cannot shadow a real one, and it stands in for "the next vendor to arrive".
_FUTURE = RobotVendor.ROS2

_SHIPPED_ROBOT_YAML = (
    Path(__file__).resolve().parents[1] / "config" / "robot" / "robot.yaml"
)


class _FutureVendorArm(DummyRobotArm):
    """A driver for a vendor with no built-in limit table, wired the way every real driver is.

    Inherits the dummy's motion surface (none of it is exercised here) and overrides the two things
    the build-time check reads: what the arm says it IS, and what pipeline it says it runs.
    """

    def __init__(
        self,
        *,
        joint_limits: JointLimitSafetyConfig | None = None,
        pipeline: SafetyPreflight | None = None,
    ) -> None:
        super().__init__()
        self._caps = RobotCapabilities(
            vendor=str(_FUTURE.value), model="fr-42", dof=6,
            supports_joint_move=True, supports_linear_move=True,
            supports_async_move=False,
            has_native_fk=False, has_native_ik=False,
            has_force_control=False, is_simulated=False,
        )
        if pipeline is None:
            pipeline = SafetyPreflight([JointLimitGuard(joint_limits or JointLimitSafetyConfig())])
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
            vendor=str(_FUTURE.value), model="fr-42", dof=6,
            supports_joint_move=True, supports_linear_move=True,
            supports_async_move=False,
            has_native_fk=False, has_native_ik=False,
            has_force_control=False, is_simulated=False,
        )


class ANewVendorWithNoLimitTableIsRefusedAtBuild(unittest.TestCase):
    """The chokepoint is ``create_arm``, because that is the one line every cell boot passes."""

    def _register(self, factory) -> None:
        register_arm_driver(_FUTURE, overwrite=True)(factory)
        self.addCleanup(unregister_arm_driver, _FUTURE)

    def test_no_table_anywhere_refuses_the_build(self) -> None:
        self._register(lambda **_: _FutureVendorArm(joint_limits=JointLimitSafetyConfig()))
        with self.assertRaises(JointLimitTableMissing) as caught:
            create_arm(_FUTURE)
        message = str(caught.exception)
        self.assertIn("min_deg", message, f"the refusal must name the key that fixes it: {message}")
        self.assertIn("max_deg", message, f"the refusal must name the key that fixes it: {message}")
        self.assertIn(
            str(_FUTURE.value), message,
            f"the refusal must name the vendor whose table is missing: {message}",
        )

    def test_static_limits_in_config_build_the_same_arm(self) -> None:
        """The control. Without it the test above would also pass for a check that refuses everything."""
        supplied = JointLimitSafetyConfig(min_deg=[-180.0] * 6, max_deg=[180.0] * 6)
        self._register(lambda **_: _FutureVendorArm(joint_limits=supplied))
        arm = create_arm(_FUTURE)
        self.assertEqual(arm.capabilities.vendor, str(_FUTURE.value))

    def test_an_arm_that_states_it_gates_nothing_still_builds(self) -> None:
        """UNGATED is a decision the operator made; this check refuses a broken guard, not no guard."""
        self._register(lambda **_: _UngatedFutureVendorArm())
        arm = create_arm(_FUTURE)
        self.assertIsNone(arm.safety_preflight)

    def test_the_guard_family_switched_off_is_not_this_defect_either(self) -> None:
        """``enforce: false`` removes the family from the pipeline, so there is no guard to starve.

        Built through ``from_safety_config`` rather than by hand, because that is the spelling every
        shipped driver uses and the omission happens THERE: the check reads the pipeline it is given,
        never the flag, so a hand-wired guard with no table is still refused whatever the flag says.
        """
        safety = RobotSafetyConfig(joint_limits=JointLimitSafetyConfig(enforce=False))
        pipeline = SafetyPreflight.from_safety_config(safety, WorkspaceLimitsConfig())
        self.assertNotIn(JointLimitGuard.name, pipeline.guard_names)
        self._register(lambda **_: _FutureVendorArm(pipeline=pipeline))
        arm = create_arm(_FUTURE)
        self.assertEqual(arm.capabilities.vendor, str(_FUTURE.value))


class TheShippedTreeIsConsistentWithWhatTheModuleClaims(unittest.TestCase):
    """Prose has no edge to the change: the docstring was RIGHT when the YAML carried numbers."""

    def test_the_module_does_not_promise_static_limits_the_shipped_yaml_ships_as_null(self) -> None:
        text = _SHIPPED_ROBOT_YAML.read_text(encoding="utf-8")
        block = re.search(r"^(\s*)joint_limits:\n((?:\1\s+.*\n|\s*\n)+)", text, re.M)
        self.assertIsNotNone(block, "the shipped robot.yaml has no safety.joint_limits block")
        assert block is not None  # for mypy; assertIsNotNone above is the real check
        body = block.group(2)
        ships_static = not re.search(r"^\s*min_deg:\s*null\s*$", body, re.M)

        doc = joint_limits_module.__doc__ or ""
        claims_the_yaml_supplies_them = bool(
            re.search(r"robot\.yaml``?\s+populates", doc)
            or re.search(r"populates\s+``?min_deg", doc),
        )
        self.assertFalse(
            claims_the_yaml_supplies_them and not ships_static,
            "src/robot/safety/joint_limits.py says the shipped robot.yaml populates "
            "min_deg / max_deg 'so the guard never fails closed on a clean install'; the shipped "
            "config/robot/robot.yaml ships both as null, so on a clean install the "
            "guard fails closed on EVERY motion for any vendor with no built-in table.",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
