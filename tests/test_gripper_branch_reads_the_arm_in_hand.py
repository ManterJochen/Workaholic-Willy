"""The gripper branch must decide from the arm it is holding, not from the arm the config names.

``from_robot_config`` lets a caller supply a live arm handle, and a supplied handle REPLACES the
construction step: from that point on the config no longer describes the arm in use. Two of the three
handle-sensitive gripper branches already knew this. Vacuum and jaw_io ask
``isinstance(arm, SupportsDigitalIO)``, so the object in hand answers the question.

MEASURED, and this is what the file pins: the Robotiq branch asked ``robot_cfg.vendor`` instead. A
``DummyRobotArm`` handed to the shipped ``vendor: ur`` tree produced ``arm=DummyRobotArm`` beside
``gripper=GripperController`` with ``substitution=None`` -- a real Robotiq driver aimed at
``robot.ur.ip`` (schema default ``192.168.1.100``, a plausible controller address on a real subnet)
while the arm held no controller connection at all, and nothing on the built cell said so.

The second class of test here guards the path that must NOT change: a real UR cell that supplies no
arm still builds a real Robotiq, because there the arm was built from this same tree and IS a UR.

The third class pins the ADDRESS half, which the vendor half deliberately left on the config: a
Robotiq is reached over the socket the URCap opens at the UR controller's address, and a non-UR tree
leaves ``robot.ur.ip`` at the schema default. ``URRobotArm`` keeps the tree it was built from, so a
supplied UR arm carries that address itself and the branch no longer has to ask a tree that does not
describe the arm. An arm that says ``ur`` and holds no address is refused rather than aimed at
whatever the tree happens to say.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config.schema.robot import RobotConfig
from src.robot.core import RobotCapabilities, RobotVendor
from src.robot.drivers import create_arm
from src.robot.execution.runtime_pick import RuntimePickService
from src.robot.grippers import SubstitutionReason
from src.robot.grippers.null import NullGripper
from src.robot.grippers.robotiq import GripperController
from tests.test_grasping_config_wiring import _calc_and_perception

#: A UR arm CONSTRUCTS without the ur_rtde SDK (the import is deferred to connect(), which no test
#: here reaches), but the startup readiness gate refuses first. Patching it off is what lets the
#: gripper branch run against a real URRobotArm on a box that has no SDK installed.
_READY = "src.robot.drivers.doctor.require_arm_vendor_ready"

_UR_CELL = {
    "vendor": "ur",
    "gripper": {"vendor": "robotiq"},
    "ur": {"ip": "10.9.9.9"},
    "grasping": {"default_mode": "auto", "max_attempts": 5},
}

#: A non-UR tree that names no controller at all, so ``ur.ip`` sits at the schema default
#: 192.168.1.100. Aiming a real Robotiq driver at THAT address is what kept the config half of this
#: branch; the arm in hand is where the address comes from instead.
_DUMMY_CELL = {
    "vendor": "dummy",
    "gripper": {"vendor": "robotiq"},
    "grasping": {"default_mode": "auto", "max_attempts": 5},
}

#: The tree the SUPPLIED arm was built from, at an address no other tree in this file names. An
#: assertion on 10.7.7.7 can only pass by reading the arm.
_ARM_OWN_CELL = {**_UR_CELL, "ur": {"ip": "10.7.7.7"}}


class _ArmThatClaimsUrAndHoldsNoAddress:
    """Says ``ur`` and carries no config, which is the one shape with no address anywhere on it.

    Not a hypothetical shape: every handle-reading branch in ``from_robot_config`` reaches through
    ``getattr``, so a partial double, an adapter around a driver, or a future UR-family driver that
    does not keep its tree all land here.
    """

    capabilities = RobotCapabilities(vendor="ur", model="ur5e")


def _substitution(gripper: object):
    substitution = getattr(gripper, "substitution", None)
    assert substitution is not None, "the cell degraded and recorded no reason"
    return substitution


def _service(config: dict, **handles: object) -> RuntimePickService:
    calc, perception = _calc_and_perception()
    return RuntimePickService.from_robot_config(
        RobotConfig(**config),  # type: ignore[arg-type]
        calculator=calc,  # type: ignore[arg-type]
        perception=perception,  # type: ignore[arg-type]
        **handles,  # type: ignore[arg-type]
    )


class ASuppliedArmDecidesTheRobotiqBranchTests(unittest.TestCase):
    """The defect: the branch read the config while an unrelated arm was in use."""

    def test_a_dummy_arm_in_hand_gets_no_robotiq_pointed_at_the_configured_controller(self) -> None:
        gripper = _service(_UR_CELL, arm=create_arm(RobotVendor.DUMMY)).orchestrator.gripper

        self.assertNotIsInstance(
            gripper,
            GripperController,
            "a dummy arm was handed in and the cell still built a real Robotiq driver aimed at "
            "robot.ur.ip",
        )
        self.assertIsInstance(gripper, NullGripper)

    def test_that_degraded_cell_says_so_on_the_object_and_not_only_in_a_log(self) -> None:
        """A warning needs a human at a terminal. The operator console has neither."""

        gripper = _service(_UR_CELL, arm=create_arm(RobotVendor.DUMMY)).orchestrator.gripper

        substitution = getattr(gripper, "substitution", None)
        self.assertIsNotNone(substitution, "the cell degraded and recorded no reason")
        assert substitution is not None
        self.assertIs(substitution.reason, SubstitutionReason.ROBOTIQ_NEEDS_UR)
        self.assertEqual(substitution.requested, "robotiq")
        # Both halves are for an operator: what is wrong, and what to do about it.
        self.assertIn("dummy", substitution.detail)
        self.assertTrue(substitution.fix.strip(), "the substitution states no fix")

    def test_a_ur_arm_in_hand_keeps_its_robotiq(self) -> None:
        """The handle test must accept the arm it was written for, not refuse every handle."""

        with patch(_READY):
            arm = create_arm(RobotVendor.UR, config=RobotConfig(**_UR_CELL))  # type: ignore[arg-type]
            gripper = _service(_UR_CELL, arm=arm).orchestrator.gripper

        assert isinstance(gripper, GripperController)
        self.assertEqual(gripper.ip, "10.9.9.9")


class TheRobotiqTakesTheAddressOfTheArmInHandTests(unittest.TestCase):
    """A real UR arm in hand answers BOTH questions: it is a UR, and it knows which controller."""

    def test_a_ur_arm_on_a_non_ur_tree_builds_a_robotiq_at_the_arms_own_address(self) -> None:
        """The case the vendor half used to refuse: the arm is real, only the tree disagrees."""

        with patch(_READY):
            arm = create_arm(RobotVendor.UR, config=RobotConfig(**_ARM_OWN_CELL))  # type: ignore[arg-type]
            gripper = _service(_DUMMY_CELL, arm=arm).orchestrator.gripper

        assert isinstance(gripper, GripperController), (
            "a real UR arm was handed in and the cell still substituted a NullGripper"
        )
        self.assertEqual(gripper.ip, "10.7.7.7")
        self.assertNotEqual(
            gripper.ip,
            RobotConfig(**_DUMMY_CELL).ur.ip,  # type: ignore[arg-type]
            "the address came from a tree that describes no UR, not from the arm",
        )
        self.assertIsNone(getattr(gripper, "substitution", None))

    def test_the_arm_outranks_a_ur_tree_that_names_another_controller(self) -> None:
        """Two addresses in play, and only one of them is the controller this arm talks to."""

        with patch(_READY):
            arm = create_arm(RobotVendor.UR, config=RobotConfig(**_ARM_OWN_CELL))  # type: ignore[arg-type]
            gripper = _service(_UR_CELL, arm=arm).orchestrator.gripper

        assert isinstance(gripper, GripperController)
        self.assertEqual(gripper.ip, "10.7.7.7")

    def test_a_ur_arm_with_no_address_is_refused_rather_than_aimed_at_the_tree(self) -> None:
        with patch(_READY):
            gripper = _service(
                _UR_CELL, arm=_ArmThatClaimsUrAndHoldsNoAddress(),
            ).orchestrator.gripper

        self.assertIsInstance(gripper, NullGripper)
        substitution = _substitution(gripper)
        self.assertIs(substitution.reason, SubstitutionReason.ROBOTIQ_NEEDS_UR)
        self.assertIn("address", substitution.detail)

    def test_the_two_refusals_do_not_read_the_same(self) -> None:
        """Same reason code, two different causes; the detail is the only place they separate."""

        with patch(_READY):
            no_address = _substitution(
                _service(_UR_CELL, arm=_ArmThatClaimsUrAndHoldsNoAddress()).orchestrator.gripper
            ).detail
        wrong_vendor = _substitution(
            _service(_UR_CELL, arm=create_arm(RobotVendor.DUMMY)).orchestrator.gripper
        ).detail

        self.assertNotEqual(no_address, wrong_vendor)
        self.assertIn("dummy", wrong_vendor)
        self.assertNotIn("dummy", no_address)


class TheRealUrCellIsUnchangedTests(unittest.TestCase):
    """The path that must not change: no handle at all, so the arm came from this same tree."""

    def test_a_ur_cell_that_supplies_no_arm_still_builds_a_robotiq(self) -> None:
        with patch(_READY):
            service = _service(_UR_CELL)

        arm = service.orchestrator.arm
        gripper = service.orchestrator.gripper
        self.assertEqual(arm.capabilities.vendor, "ur")
        assert isinstance(gripper, GripperController)
        self.assertEqual(gripper.ip, "10.9.9.9")
        self.assertIsNone(getattr(gripper, "substitution", None))

    def test_a_non_ur_cell_that_supplies_no_arm_still_substitutes(self) -> None:
        """Unchanged too, and by the same rule: the arm this tree builds is not a UR.

        The tree here does name a controller (``ur.ip: 10.9.9.9``, carried over from ``_UR_CELL``)
        and it buys nothing, which is the point: an address is not a UR arm to reach it on.
        """

        gripper = _service({**_UR_CELL, "vendor": "dummy"}).orchestrator.gripper

        self.assertIsInstance(gripper, NullGripper)
        substitution = _substitution(gripper)
        self.assertIs(substitution.reason, SubstitutionReason.ROBOTIQ_NEEDS_UR)
        # Named for the arm that was built, not for the tree that named it.
        self.assertIn("dummy", substitution.detail)


if __name__ == "__main__":
    unittest.main()
