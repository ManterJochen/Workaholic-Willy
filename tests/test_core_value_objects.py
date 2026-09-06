"""Gap L0.11 (+ L0.7) — tests for the robot.core contract value objects.

Covers RobotVendor/GripperVendor.from_string (valid / case-insensitive / unknown), RobotCapabilities
validation, the RobotError hierarchy, and JointPositions.check_dof (the new canonical DoF check).
"""

from __future__ import annotations

import pytest

from src.robot.core import (
    GripperVendor,
    JointPositions,
    RobotCapabilities,
    RobotVendor,
)
from src.robot.core.errors import (
    RobotConnectionError,
    RobotError,
    RobotKinematicsError,
)


class TestVendorEnums:
    def test_from_string_valid_and_case_insensitive(self) -> None:
        assert RobotVendor.from_string("ur") is RobotVendor.UR
        assert RobotVendor.from_string("UR") is RobotVendor.UR
        assert RobotVendor.from_string("Sim") is RobotVendor.SIM
        assert GripperVendor.from_string("robotiq") is GripperVendor.ROBOTIQ
        assert GripperVendor.from_string("ROBOTIQ") is GripperVendor.ROBOTIQ

    def test_from_string_unknown_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            RobotVendor.from_string("ur5")
        with pytest.raises((ValueError, KeyError)):
            GripperVendor.from_string("not_a_gripper")


class TestRobotCapabilities:
    def test_valid_construct(self) -> None:
        cap = RobotCapabilities(vendor="ur", dof=6)
        assert cap.vendor == "ur"
        assert cap.dof == 6

    def test_empty_vendor_raises(self) -> None:
        with pytest.raises(ValueError):
            RobotCapabilities(vendor="")

    def test_non_lowercase_or_spaced_vendor_raises(self) -> None:
        with pytest.raises(ValueError):
            RobotCapabilities(vendor="UR")
        with pytest.raises(ValueError):
            RobotCapabilities(vendor="my vendor")

    def test_non_positive_dof_raises(self) -> None:
        with pytest.raises(ValueError):
            RobotCapabilities(vendor="ur", dof=0)


class TestRobotErrorHierarchy:
    def test_subclass_relationships(self) -> None:
        assert issubclass(RobotError, Exception)
        for cls in (RobotConnectionError, RobotKinematicsError):
            assert issubclass(cls, RobotError)

    def test_caught_as_base(self) -> None:
        with pytest.raises(RobotError):
            raise RobotConnectionError("boom")


class TestJointPositionsCheckDof:
    def test_check_dof_passes_and_returns_self(self) -> None:
        jp = JointPositions([0.0] * 6)
        assert jp.dof == 6
        assert jp.check_dof(6) is jp

    def test_check_dof_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            JointPositions([0.0] * 6).check_dof(7)
