"""The arm says who it is: model, joints, home and controller come from its description and its tree (P0e E4).

Until P0e the UR driver built its identity out of constants: a capability record whose `dof` defaulted to six and whose
model was replaced in afterwards, a home pose authored for a UR5e, a joint-name tuple inside the planner, and a lock key
and a Robotiq address read back off the tree the arm keeps. Each is now read from the description of the model the tree
names, or from the arm's own `RobotCapabilities`. That is what a second vendor needs: a Franka has seven joints, another
home and another joint order, and keeps no UR block an address could be read from.
"""

from __future__ import annotations

import math
import unittest

from src.config.arms import admitted_arms, load_arm
from src.config.schema.robot import RobotConfig
from src.robot.core import RobotCapabilities
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.execution.cell_lock import LockKeyRequired, cell_lock_key
from src.robot.execution.robot import Robot


def _ur(model: str = "ur5e", ip: str | None = "10.1.2.3", **extra: object) -> URRobotArm:
    tree: dict[str, object] = {"vendor": "ur", "ur": {"model": model, **({"ip": ip} if ip else {})},
                               "gripper": {"model": "robotiq_2f85"}, **extra}
    return URRobotArm(RobotConfig.model_validate(tree))


class _Stated:
    """An arm double that keeps no tree at all and states its identity, the way a new vendor's driver will."""

    def __init__(self, capabilities: RobotCapabilities) -> None:
        self.capabilities = capabilities


class TheUrArmReadsItsDescriptionTests(unittest.TestCase):
    def test_every_admitted_model_reports_what_its_description_and_its_tree_say(self) -> None:
        for model in admitted_arms():
            with self.subTest(model=model):
                arm, description = _ur(model), load_arm(model)
                caps = arm.capabilities
                self.assertEqual((caps.vendor, caps.model, caps.dof, caps.controller, caps.owns_a_controller),
                                 ("ur", model, description.dof, "10.1.2.3", True))
                self.assertEqual(arm._home_joints, [joint.home_rad for joint in description.joints])
                self.assertEqual(arm._joint_names, tuple(joint.name for joint in description.joints))

    def test_the_home_is_the_pose_it_always_was(self) -> None:
        """The literal the deleted HOME_JOINTS_DEFAULT held, so the move to the description changed no pose.

        Shoulder lift and wrist 1 at -pi/2, the rest at zero. The constant's own comment put the second -pi/2 on
        wrist 2; the tuple and every description put it on wrist 1, and the tuple is what the driver sent.
        """
        half = -math.pi / 2
        self.assertEqual(_ur()._home_joints, [0.0, half, 0.0, half, 0.0, 0.0])

    def test_a_home_the_tree_declares_outranks_the_description(self) -> None:
        declared_deg = [5.0, -70.0, 75.0, -80.0, -85.0, 35.0]
        self.assertEqual(_ur(motion={"home_deg": declared_deg})._home_joints, [math.radians(v) for v in declared_deg])

    def test_an_unaddressed_cell_owns_a_controller_and_names_none(self) -> None:
        caps = _ur(ip=None).capabilities
        self.assertEqual((caps.controller, caps.owns_a_controller), (None, True))

    def test_the_planner_is_handed_the_description_joint_order(self) -> None:
        arm = _ur(motion={"planner": "curobo"}, safety={"self_collision": {"planner_margin_mm": 4.0}})
        arm._curobo_client_factory = lambda: None  # type: ignore[assignment,return-value]
        self.assertEqual(arm._curobo_planner()._joint_names, tuple(j.name for j in load_arm("ur5e").joints))


class TheLockReadsTheArmsIdentityTests(unittest.TestCase):
    def test_an_arm_that_keeps_no_tree_is_locked_on_the_controller_it_names(self) -> None:
        """The new reach: before P0e an adapter that kept no tree could be refused, never locked."""
        arm = _Stated(RobotCapabilities(vendor="franka", model="fr3", dof=7, controller="172.16.0.2",
                                        owns_a_controller=True))
        self.assertEqual(Robot.from_parts(arm=arm, gripper=None).lock_key, "franka@172.16.0.2")  # type: ignore[arg-type]

    def test_an_arm_that_drives_a_controller_and_names_none_is_refused(self) -> None:
        arm = _Stated(RobotCapabilities(vendor="franka", model="fr3", dof=7, controller=None, owns_a_controller=True))
        with self.assertRaisesRegex(LockKeyRequired, r"'franka'.*lock_key=None"):
            Robot.from_parts(arm=arm, gripper=None)  # type: ignore[arg-type]

    def test_an_arm_that_drives_no_controller_takes_no_lock(self) -> None:
        arm = _Stated(RobotCapabilities(vendor="sim", model="ur5e", dof=6, controller=None, owns_a_controller=False))
        self.assertIsNone(Robot.from_parts(arm=arm, gripper=None).lock_key)  # type: ignore[arg-type]

    def test_a_ur_arm_is_locked_on_the_key_its_tree_derives(self) -> None:
        """The two derivations agree on the arm both can read, so the move changed no key."""
        arm = _ur(ip="10.9.9.9")
        self.assertEqual(Robot.from_parts(arm=arm, gripper=None).lock_key, cell_lock_key(arm.config))
        self.assertEqual(cell_lock_key(arm.config), "ur@10.9.9.9")


class TheIdentityIsStatedTests(unittest.TestCase):
    def test_no_field_of_the_identity_has_a_default(self) -> None:
        for missing in ("dof", "controller", "owns_a_controller"):
            given = {"vendor": "ur", "dof": 6, "controller": None, "owns_a_controller": True}
            del given[missing]
            with self.subTest(missing=missing), self.assertRaises(TypeError):
                RobotCapabilities(**given)  # type: ignore[arg-type]

    def test_an_address_on_an_arm_that_drives_no_controller_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "owns none"):
            RobotCapabilities(vendor="sim", dof=6, controller="127.0.0.1", owns_a_controller=False)

    def test_a_blank_address_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "blank"):
            RobotCapabilities(vendor="ur", dof=6, controller=" ", owns_a_controller=True)


if __name__ == "__main__":
    unittest.main()
