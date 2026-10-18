"""Every registered arm driver meets the ``RobotArm`` Protocol member for member, return type for return type.

``@runtime_checkable`` validates member *presence* only, so a driver could drift a return type undetected; this
locks them, as ``tests/robot/grippers/test_gripper_protocol_conformance.py`` does for grippers. Since P0g slice 10 it is driven by
the registry and by the Protocol itself: the drivers are every vendor ``available_vendors()`` lists, built through
``tests/_registered_arms.py``, and the members are what ``RobotArm`` declares, introspected. The hand-kept driver
list it had would have missed a fifth vendor, and the hand-kept member list had to be edited for every new verb.
"""

from __future__ import annotations

import inspect
import typing
import unittest

from src.robot.core.robot_arm import RobotArm
from tests import _registered_arms


def _members() -> tuple[str, ...]:
    return tuple(sorted(typing._get_protocol_attrs(RobotArm)))  # type: ignore[attr-defined]


def _return_hint(cls: type, name: str) -> object:
    attr = inspect.getattr_static(cls, name)
    func = attr.fget if isinstance(attr, property) else attr
    return typing.get_type_hints(func).get("return")


def _missing(registered: list[str], table: tuple[str, ...]) -> list[str]:
    """Registered vendors the table cannot build."""
    return sorted(set(registered) - set(table))


class TheTableIsTheRegistryTests(unittest.TestCase):
    def test_the_table_is_the_registry(self) -> None:
        from src.robot.drivers import available_vendors

        registered = available_vendors()
        self.assertEqual([], _missing(registered, _registered_arms.vendors()),
                         "a registered vendor tests cannot build: add its row to tests/_registered_arms.py")
        self.assertEqual([], sorted(set(_registered_arms.vendors()) - set(registered)), "a row for no vendor")

    def test_a_vendor_the_table_does_not_know_is_found(self) -> None:
        """⭐ THE POSITIVE CONTROL: a fifth vendor registered without a row is named."""
        self.assertEqual(["franka"], _missing(["dummy", "franka", "sim", "ur"], _registered_arms.vendors()))

    def test_the_members_are_the_protocols(self) -> None:
        """⭐ The introspection sees what the Protocol declares: a blind one would pass every driver."""
        members = _members()
        for name in ("move", "move_to_joints", "connect", "capabilities"):
            self.assertIn(name, members)


class RobotArmReturnTypeConformanceTests(unittest.TestCase):
    def test_every_registered_arm_matches_the_protocols_return_types(self) -> None:
        for vendor in _registered_arms.vendors():
            driver = type(_registered_arms.build(vendor))
            for name in _members():
                expected = _return_hint(RobotArm, name)
                with self.subTest(vendor=vendor, member=name):
                    self.assertIsNotNone(expected, f"RobotArm Protocol member {name} has no return hint")
                    self.assertEqual(_return_hint(driver, name), expected,
                                     f"{driver.__name__}.{name} return type drifted from the RobotArm Protocol")

    def test_every_registered_arm_has_every_protocol_member(self) -> None:
        for vendor in _registered_arms.vendors():
            driver = type(_registered_arms.build(vendor))
            for name in _members():
                with self.subTest(vendor=vendor, member=name):
                    self.assertTrue(hasattr(driver, name), f"{driver.__name__} is missing RobotArm member {name}")


if __name__ == "__main__":
    unittest.main()
