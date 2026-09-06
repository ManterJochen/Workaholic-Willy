"""Signature-level conformance of every registered RobotArm driver to the
RobotArm Protocol's return-type contract.

``@runtime_checkable`` only validates member *presence*, so a driver could drift a
return type undetected. This locks the return types of all four drivers — the same
guard the Gripper Protocol got in Iteration 2 (tests/test_gripper_protocol_conformance.py).
"""

from __future__ import annotations

import inspect
import typing
import unittest

from src.robot.core.robot_arm import RobotArm
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.kuka.arm import KukaRobotArm
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.ur.arm import URRobotArm

_DRIVERS = (URRobotArm, KukaRobotArm, IsaacRobotArm, DummyRobotArm)

_PROTOCOL_MEMBERS = (
    "capabilities",
    "is_connected",
    "connect",
    "disconnect",
    "get_tcp_pose",
    "get_joint_positions",
    "move_joint",
    "move_linear",
    "stop",
    "fk",
    "ik",
    "is_inside_workspace",
    "move_to",
    "move_home",
    "wait_until_steady",
    "move",
)


def _return_hint(cls: type, name: str) -> object:
    attr = inspect.getattr_static(cls, name)
    func = attr.fget if isinstance(attr, property) else attr
    return typing.get_type_hints(func).get("return")


class RobotArmReturnTypeConformanceTests(unittest.TestCase):
    def test_every_driver_matches_protocol_return_types(self) -> None:
        for name in _PROTOCOL_MEMBERS:
            expected = _return_hint(RobotArm, name)
            self.assertIsNotNone(
                expected, f"RobotArm Protocol member {name} has no return hint"
            )
            for driver in _DRIVERS:
                with self.subTest(member=name, driver=driver.__name__):
                    self.assertEqual(
                        _return_hint(driver, name),
                        expected,
                        f"{driver.__name__}.{name} return type drifted from the "
                        f"RobotArm Protocol",
                    )

    def test_every_driver_has_all_protocol_members(self) -> None:
        # The presence check @runtime_checkable performs, made explicit per driver.
        for driver in _DRIVERS:
            for name in _PROTOCOL_MEMBERS:
                with self.subTest(driver=driver.__name__, member=name):
                    self.assertTrue(
                        hasattr(driver, name),
                        f"{driver.__name__} is missing RobotArm member {name}",
                    )


if __name__ == "__main__":
    unittest.main()
