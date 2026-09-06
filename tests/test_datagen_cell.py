"""Pointing datagen at a cell that is not the owner's.

⭐ WHAT THIS IS FOR. The generator described ONE physical setup: two obliques at a mirrored offset,
an overhead, one wrist view, all sharing a resolution and a field of view, all aimed at the same
point. Every one of those is a property of that cell, and a customer with four fixed cameras, or two
that are not mirror images, or a calibrated extrinsic they measured rather than derived, could
express none of it.

⛔⛔ AND THE OLD RIG HAD A TRAP THAT PUNISHED THE OBVIOUS FIX. `CameraRigConfig.views` was four
literal names, and `scenes/layout.py` decided each camera's MOUNT by comparing that name: `overhead`
and the two obliques became FIXED, and the `else` branch made everything else a WRIST camera. So the
natural way to add a fourth fixed camera -- widen the name list -- produced an eye-in-hand camera the
renderer would then try to drive the arm to.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.config import CameraRigConfig, CameraSpec, DatagenConfig
from datagen.scenes.layout import _cameras
from datagen.scenes.spec import CameraMount


class TheBuiltInRigIsUnchangedTests(unittest.TestCase):
    """⚠ THE SEAM IS ADDITIVE AND DEFAULT-OFF. A run that declares no cameras must be what it was."""

    def test_the_default_declares_no_cameras(self) -> None:
        rig = CameraRigConfig()
        self.assertIsNone(rig.cameras)
        self.assertFalse(rig.allow_unmounted_wrist)

    def test_the_default_rig_still_builds_its_three_views(self) -> None:
        built = _cameras(DatagenConfig(), np.random.default_rng(0))          # noqa: SLF001
        self.assertEqual(["wrist", "oblique_left", "oblique_right"], [c.name for c in built])
        self.assertIs(CameraMount.WRIST, built[0].mount)
        self.assertIs(CameraMount.FIXED, built[1].mount)


class ADeclaredRigIsReadTests(unittest.TestCase):

    @staticmethod
    def _cell(**rig: object) -> DatagenConfig:
        """⭐ THE ARM FOLLOWS THE CAMERAS, and the fixture derives it rather than pinning one.

        Two rules meet here and they are each other's mirror: a wrist camera with no arm is refused,
        and a posed arm with no wrist camera is refused. Together they say the arm and the
        eye-in-hand view go together, so a fixture that hard-codes either one is wrong for half the
        cases in this class. Both rules fired on the first version of this helper, in opposite
        directions, which is the pair working rather than the fixtures needing an exception.
        """
        cameras = rig.get("cameras") or ()
        wants_arm = any(getattr(c, "mount", "fixed") == "wrist" for c in cameras)
        return DatagenConfig(camera_rig=rig,
                             render={"arm": {"mode": "posed" if wants_arm else "absent"}})

    def test_a_customer_may_have_more_fixed_cameras_than_the_built_in_rig_allowed(self) -> None:
        """Three was the ceiling, because three names mapped to FIXED and every other name did not."""
        cell = self._cell(cameras=(
            CameraSpec(name="a", position_mm=(0.0, -400.0, 700.0)),
            CameraSpec(name="b", position_mm=(0.0, 400.0, 700.0)),
            CameraSpec(name="c", position_mm=(-500.0, 0.0, 500.0)),
            CameraSpec(name="d", position_mm=(500.0, 0.0, 500.0)),
        ))
        built = _cameras(cell, np.random.default_rng(0))                      # noqa: SLF001
        self.assertEqual(4, sum(1 for c in built if c.mount is CameraMount.FIXED))

    def test_an_unrecognised_NAME_no_longer_becomes_a_wrist_camera(self) -> None:
        """⛔ THE DEFECT, stated as a test. Under the name dispatch a camera called `front` fell to
        the `else` branch and came out WRIST, so the renderer had to drive the arm to reach a camera
        the customer had bolted to a post."""
        cell = self._cell(cameras=(CameraSpec(name="front", position_mm=(-500.0, 0.0, 500.0)),))
        built = _cameras(cell, np.random.default_rng(0))                      # noqa: SLF001
        self.assertIs(CameraMount.FIXED, built[0].mount)

    def test_the_mount_is_declared_and_survives_the_name(self) -> None:
        """The other direction: a camera named `overhead` is WRIST if the caller says so. A name is
        a label and must not decide whether the robot has to drive somewhere."""
        cell = self._cell(cameras=(CameraSpec(name="overhead", mount="wrist"),))
        built = _cameras(cell, np.random.default_rng(0))                      # noqa: SLF001
        self.assertIs(CameraMount.WRIST, built[0].mount)

    def test_optics_are_per_camera_and_fall_back_to_the_rig(self) -> None:
        """A wide wrist camera beside narrower fixed ones used to cost a second dataset."""
        cell = self._cell(resolution=(640, 480), horizontal_fov_deg=47.0, cameras=(
            CameraSpec(name="wide", position_mm=(0.0, 0.0, 900.0),
                       resolution=(1280, 720), horizontal_fov_deg=69.0),
            CameraSpec(name="plain", position_mm=(0.0, 0.0, 800.0)),
        ))
        built = _cameras(cell, np.random.default_rng(0))                      # noqa: SLF001
        self.assertEqual(((1280, 720), 69.0), (built[0].resolution, built[0].horizontal_fov_deg))
        self.assertEqual(((640, 480), 47.0), (built[1].resolution, built[1].horizontal_fov_deg))

    def test_a_wrist_camera_may_omit_its_position_and_is_then_sampled(self) -> None:
        cell = self._cell(cameras=(CameraSpec(name="eih", mount="wrist"),))
        first = _cameras(cell, np.random.default_rng(1))[0]                   # noqa: SLF001
        second = _cameras(cell, np.random.default_rng(2))[0]                  # noqa: SLF001
        self.assertNotEqual(first.position_mm, second.position_mm)

    def test_a_fixed_camera_without_a_position_is_refused(self) -> None:
        """There is nothing to derive it from, and guessing one would put a camera somewhere the
        customer never placed it."""
        with self.assertRaises(Exception) as caught:
            CameraSpec(name="nowhere", mount="fixed")
        self.assertIn("position_mm", str(caught.exception))

    def test_duplicate_names_are_refused(self) -> None:
        """The name is the image filename prefix and the key in the view record, so two cameras
        sharing one would overwrite each other's images."""
        with self.assertRaises(Exception):
            CameraRigConfig(cameras=(CameraSpec(name="a", position_mm=(0.0, 0.0, 1.0)),
                                     CameraSpec(name="a", position_mm=(0.0, 0.0, 2.0))))

    def test_an_empty_declared_list_is_refused_rather_than_falling_back(self) -> None:
        """`None` means "use the built-in rig"; an empty tuple means the caller meant to declare and
        declared nothing, which is a scene with no view."""
        with self.assertRaises(Exception):
            CameraRigConfig(cameras=())


class EyeInHandBelongsOnTheArmTests(unittest.TestCase):
    """⛔ OWNER DECISION, 2026-09-04: possible, but it has to be asked for.

    `render/isaac.py:1409` places a wrist viewpoint with no arm in the scene when `arm.mode` is
    `absent`, via a bare `pass`, arguing that the pose is a valid camera pose and the dataset records
    the arm mode. Both are true. It is still almost never what the caller meant: the arm is this
    cell's dominant occluder, so its absence changes every image rather than decorating it.
    """

    def test_an_ASKED_FOR_wrist_view_with_no_arm_is_refused_at_CONFIG_time(self) -> None:
        """Config time is the point. The old path produced a whole dataset and recorded the
        disagreement in a field somebody had to think to read."""
        with self.assertRaises(Exception) as caught:
            DatagenConfig(render={"arm": {"mode": "absent"}},
                          camera_rig={"views": ("wrist", "oblique_left")})
        message = str(caught.exception)
        self.assertIn("wrist", message)
        self.assertIn("allow_unmounted_wrist", message, "the refusal must name the way out")

    def test_the_DEFAULTED_wrist_view_is_not_treated_as_a_request(self) -> None:
        """⚠ THE BOUNDARY, AND IT IS DELIBERATE. `views` defaults to the owner's three-view rig,
        which has a wrist camera because the owner's cell has an arm. Someone who writes
        `arm.mode = "absent"` and leaves `views` alone is saying "no robot", not "give me an
        eye-in-hand view without one", and refusing them would make the natural way to ask for
        armless tabletop data an error. Six existing tests said so before this boundary existed.

        ⛔ SO THIS CASE STILL RENDERS A FREE-FLOATING WRIST VIEW, exactly as before. The rule
        narrows the failure to the case somebody meant; it does not remove the behaviour."""
        config = DatagenConfig(render={"arm": {"mode": "absent"}})
        self.assertIn("wrist", config.camera_rig.views)

    def test_chosen_and_defaulted_are_told_apart_by_model_fields_set(self) -> None:
        """The mechanism, pinned. It is the same distinction as the sibling package's UNSET
        sentinel: chosen and defaulted are different facts, and a check that cannot tell them apart
        has to guess which one it is looking at."""
        self.assertEqual(set(), CameraRigConfig().model_fields_set)
        self.assertIn("views", CameraRigConfig(views=("wrist",)).model_fields_set)

    def test_it_can_be_asked_for_explicitly(self) -> None:
        config = DatagenConfig(render={"arm": {"mode": "absent"}},
                               camera_rig={"allow_unmounted_wrist": True})
        self.assertTrue(config.camera_rig.allow_unmounted_wrist)

    def test_a_declared_rig_is_checked_the_same_way(self) -> None:
        """The rule follows the cameras, not the four built-in names."""
        with self.assertRaises(Exception):
            DatagenConfig(render={"arm": {"mode": "absent"}},
                          camera_rig={"cameras": (CameraSpec(name="eih", mount="wrist"),)})

    def test_no_wrist_view_and_no_arm_is_fine(self) -> None:
        """The control. Without it the rule could be refusing every armless config."""
        config = DatagenConfig(render={"arm": {"mode": "absent"}},
                               camera_rig={"views": ("oblique_left", "oblique_right")})
        self.assertEqual(("oblique_left", "oblique_right"), config.camera_rig.views)

    def test_an_arm_in_the_scene_needs_no_permission(self) -> None:
        self.assertIn("wrist", DatagenConfig().camera_rig.views)


class ACustomerMayDeclareTheirOwnWristMountTests(unittest.TestCase):
    """⛔ THE GATE IN `robots.py` IS RIGHT AND STAYS. `_measured_mount` answers for `ur5e` alone,
    because its four constants came from running the calibration on that arm, and the UR e-series
    sharing a wrist-3 flange is not the same as a mount having been measured. Falling back to another
    robot's numbers would put the camera in the wrong place and record it as if it belonged there.

    What was missing is the other half: a customer who HAS run their own calibration had no way to
    say so, and their only route was editing a file inside `backend/`.
    """

    def test_a_declared_mount_reaches_the_robot_definition(self) -> None:
        from datagen.robots import declared_mount, resolve_robot

        from datagen.config import WristMountSpec

        spec = WristMountSpec(offset_mm=(10.0, 0.0, 40.0), aim_target_mm=(10.0, 0.0, 200.0),
                              source="our own run_eih_calibrate, 2026-09-04")
        definition = resolve_robot("ur3e", declared_mount(spec))
        self.assertIsNotNone(definition.wrist_camera_mount)
        self.assertTrue(definition.can_render_wrist_view)

    def test_without_one_a_non_ur5e_model_still_has_no_mount(self) -> None:
        """The control, and the gate this does not remove."""
        from datagen.robots import resolve_robot

        self.assertIsNone(resolve_robot("ur3e").wrist_camera_mount)

    def test_the_source_is_carried_rather_than_invented(self) -> None:
        """⚠ A mount is a MEASUREMENT, and a measurement without a provenance is a number. This is
        what keeps a customer's mount distinguishable from the one this repository measured."""
        from datagen.robots import declared_mount

        from datagen.config import WristMountSpec

        mount = declared_mount(WristMountSpec(
            offset_mm=(0.0, 0.0, 0.0), aim_target_mm=(0.0, 0.0, 100.0), source="measured on our UR3e"))
        assert mount is not None
        self.assertEqual("measured on our UR3e", mount.source)

    def test_no_declaration_means_no_change(self) -> None:
        from datagen.robots import declared_mount

        self.assertIsNone(declared_mount(None))


if __name__ == "__main__":
    unittest.main()
