"""The robot contract: what the generator will place, and what it refuses to guess.

No Isaac and no GPU — the whole point is that a robot is rejected on a laptop, in a second, with every
missing fact named, rather than after an Isaac boot or (worse) after a dataset has been trained on.
"""

from __future__ import annotations

import unittest

from src.robot.drivers.sim.robot_models import _UR_MODELS
from datagen.robots import (
    RobotNotUsable,
    resolve_robot,
    workspace_overshoot_mm,
)


class RegistryTests(unittest.TestCase):
    def test_registry_workspaces_are_inside_reach(self) -> None:
        """Every stated workspace must actually fit the model that states it.

        This is the test that earns the right to write the workspace out explicitly instead of deriving
        it. It has already paid for itself: the first UR3e value (330 +- 110) overshot by 18.3 mm and
        was caught here, not by a dataset of unpickable scenes.
        """
        for key in _UR_MODELS:
            with self.subTest(robot=key):
                definition = resolve_robot(key)
                overshoot = workspace_overshoot_mm(
                    definition, definition.workspace_center_mm, definition.workspace_half_extents_mm,
                )
                self.assertEqual(
                    overshoot, 0.0,
                    f"{key}: its own registry workspace "
                    f"{definition.workspace_center_mm}+-{definition.workspace_half_extents_mm} "
                    f"overshoots the reach sphere by {overshoot:.1f} mm",
                )

    def test_the_ur5e_patch_does_not_fit_a_ur3e(self) -> None:
        """The whole reason workspaces are per-robot, as a number rather than a claim."""
        ur3e = resolve_robot("ur3e")
        overshoot = workspace_overshoot_mm(ur3e, (450.0, 0.0), (150.0, 150.0))
        self.assertGreater(overshoot, 100.0, "a UR3e should not be able to reach the UR5e pick patch")

    def test_a_bare_model_gets_a_gripper_attached(self) -> None:
        """ur3e.usd ships with no gripper; both datasets must still show the same occluder."""
        self.assertTrue(resolve_robot("ur5e").gripper_is_baked)
        self.assertFalse(resolve_robot("ur3e").gripper_is_baked)
        self.assertTrue(resolve_robot("ur3e").gripper)

    def test_only_a_measured_wrist_mount_enables_the_wrist_view(self) -> None:
        """Sharing a flange is not the same as having been measured on it."""
        self.assertTrue(resolve_robot("ur5e").can_render_wrist_view)
        self.assertIsNone(resolve_robot("ur3e").wrist_camera_mount)
        self.assertFalse(resolve_robot("ur3e").can_render_wrist_view)

    def test_the_measured_mount_carries_its_provenance(self) -> None:
        mount = resolve_robot("ur5e").wrist_camera_mount
        assert mount is not None
        self.assertIn("measured", mount.source)
        self.assertEqual(mount.offset_mm, (130.0, 20.0, 0.0))

    def test_an_unknown_robot_is_refused_with_a_recipe(self) -> None:
        """A refusal that does not say how to fix it just moves the work to the reader."""
        with self.assertRaises(RobotNotUsable) as ctx:
            resolve_robot("franka_panda")
        message = str(ctx.exception)
        for expected in ("URModelSpec", "UR_DH_TABLES_M", "cuRobo", "run_eih_calibrate", "verify-robot"):
            self.assertIn(expected, message, f"the refusal should tell the reader about {expected}")

    def test_every_registered_robot_resolves_completely(self) -> None:
        """There is no half-valid definition: resolve_robot returns a whole one or raises."""
        for key in _UR_MODELS:
            with self.subTest(robot=key):
                definition = resolve_robot(key)
                self.assertGreater(definition.max_reach_mm, 0.0)
                self.assertGreater(definition.shoulder_height_mm, 0.0)
                self.assertTrue(definition.usd_relpath.endswith(".usd"))
                self.assertTrue(definition.wrist_link_name)


class ConfigRefusalTests(unittest.TestCase):
    """The refusal happens at CONFIG time — the earliest point, before a GPU is involved."""

    def test_the_default_config_places_an_arm_and_is_valid(self) -> None:
        """The default used to be un-renderable: mode='posed' while the renderer only did 'absent'."""
        from datagen.config import DatagenConfig

        config = DatagenConfig()
        self.assertEqual(config.render.arm.mode, "posed")
        self.assertEqual(config.render.arm.robot_model, "ur5e")

    def test_a_ur3e_is_refused_the_ur5e_workspace_with_a_usable_suggestion(self) -> None:
        from datagen.config import DatagenConfig

        with self.assertRaises(ValueError) as ctx:
            DatagenConfig(render={"arm": {"mode": "posed", "robot_model": "ur3e"}})
        message = str(ctx.exception)
        self.assertIn("177 mm beyond", message)
        self.assertIn("(300.0, 0.0)", message, "the refusal must name a workspace that DOES fit")
        self.assertTrue(message.isascii(), "printed to a cp1252 console; non-ASCII arrives corrupted")

    def test_a_ur3e_with_its_own_workspace_is_accepted(self) -> None:
        from datagen.config import DatagenConfig

        config = DatagenConfig(
            workspace={"center_mm": (300.0, 0.0), "half_extents_mm": (100.0, 100.0)},
            render={"arm": {"mode": "posed", "robot_model": "ur3e"}},
        )
        self.assertEqual(config.workspace.center_mm, (300.0, 0.0))

    def test_no_arm_means_no_reach_requirement(self) -> None:
        """A tabletop dataset without an arm is legitimate and has no reach to satisfy."""
        from datagen.config import DatagenConfig

        config = DatagenConfig(
            domain="tabletop",
            workspace={"center_mm": (900.0, 0.0), "half_extents_mm": (300.0, 300.0)},
            render={"arm": {"mode": "absent"}},
        )
        self.assertEqual(config.render.arm.mode, "absent")


class OvershootTests(unittest.TestCase):
    def test_the_worst_CORNER_decides_not_the_centre(self) -> None:
        """A comfortable centre with an unreachable corner is the case that must not pass."""
        ur3e = resolve_robot("ur3e")
        centre_only = workspace_overshoot_mm(ur3e, (420.0, 0.0), (0.0, 0.0))
        with_corners = workspace_overshoot_mm(ur3e, (420.0, 0.0), (120.0, 120.0))
        self.assertEqual(centre_only, 0.0)
        self.assertGreater(with_corners, 0.0)

    def test_a_workspace_is_measured_from_the_SHOULDER(self) -> None:
        """The reach sphere is centred on the shoulder, so table height changes the answer."""
        ur5e = resolve_robot("ur5e")
        at_table = workspace_overshoot_mm(ur5e, (800.0, 0.0), (10.0, 10.0), table_height_mm=0.0)
        at_shoulder = workspace_overshoot_mm(
            ur5e, (800.0, 0.0), (10.0, 10.0), table_height_mm=ur5e.shoulder_height_mm,
        )
        self.assertGreater(at_table, at_shoulder)


if __name__ == "__main__":
    unittest.main()
