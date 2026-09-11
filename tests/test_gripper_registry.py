"""A gripper is described once, as data, and four of the places that describe the 2F-85 today agree with it.

The Hand-E made the problem visible. One hand was described in six unlinked places (the labeller's
`JawModel`, the network's `JAW_GEOMETRY`, the runtime envelope `grasping.gripper_geometry`, the actuation
widths in `robot.gripper`, the sim profile, the planner bundles) under three spellings of its name, and
nothing cross-checked them. The owner decided on 2026-09-11 that one YAML file per hand is the
description and that `robot.gripper.model` names it.

This file pins the contract before anything reads it. The shipped 2F-85 file reproduces the four
statements it names, field by field, so a drift in any of those four turns it red. The other two are
not compared and do not all agree: the sim `GripperProfile` opens to 87.1 mm in Isaac against the 85.0
written here, and the planner bundles carry their own meshes. They move onto the registry in lane (i)
of the plan. The registry refuses what it cannot resolve instead of guessing.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.config import ConfigError
from src.config.grippers import available_grippers, load_gripper
from src.config.schema.robot import GripperConfig, RobotConfig
from src.config.schema.robot.grasping_schema import GraspingParallelJawGeometryConfig

_DATA = Path(__file__).resolve().parents[1] / "config"


class The2F85FileIsWhatTheCopiesSayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jaw = load_gripper("robotiq_2f85").jaw

    def test_it_is_the_labellers_jaw_model(self) -> None:
        from datagen.grasps.verdict import JawModel

        model = JawModel()
        for name in ("aperture_mm", "min_width_mm", "finger_ahead_mm", "finger_behind_mm",
                     "finger_thickness_mm", "finger_width_mm", "palm_depth_mm", "palm_width_mm",
                     "friction_coefficient", "pad_ahead_mm", "pad_behind_mm"):
            with self.subTest(field=name):
                self.assertEqual(getattr(self.jaw, name), getattr(model, name))

    def test_it_is_the_networks_conditioning_entry(self) -> None:
        from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY

        self.assertEqual({
            "aperture_mm": self.jaw.aperture_mm,
            "min_width_mm": self.jaw.min_width_mm,
            "finger_ahead_mm": self.jaw.finger_ahead_mm,
            "finger_behind_mm": self.jaw.finger_behind_mm,
            "finger_thickness_mm": self.jaw.finger_thickness_mm,
            "finger_width_mm": self.jaw.finger_width_mm,
            "pad_span_mm": self.jaw.pad_length_mm,
            "friction_coefficient": self.jaw.friction_coefficient,
        }, JAW_GEOMETRY["2f85"])

    def test_it_is_the_runtime_envelope_default(self) -> None:
        envelope = GraspingParallelJawGeometryConfig()
        for name, expected in {
            "finger_length_mm": envelope.finger_length_mm,
            "finger_thickness_mm": envelope.finger_thickness_mm,
            "finger_width_mm": envelope.finger_width_mm,
            "finger_pad_overlap_mm": envelope.finger_pad_overlap_mm,
            "finger_ahead_mm": envelope.fingertip_depth_mm,
            "pad_length_mm": envelope.pad_length_mm,
            "pad_ahead_mm": envelope.pad_ahead_mm,
            "palm_depth_mm": envelope.palm_depth_mm,
            "palm_width_mm": envelope.palm_width_mm,
        }.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(self.jaw, name), expected)

    def test_it_is_the_actuation_default(self) -> None:
        actuation = GripperConfig()
        self.assertEqual(self.jaw.aperture_mm, actuation.max_width_mm)
        self.assertEqual(self.jaw.min_width_mm, actuation.min_width_mm)
        self.assertEqual(self.jaw.closed_width_mm, actuation.closed_width_mm)

    def test_what_was_never_measured_says_so(self) -> None:
        self.assertFalse(self.jaw.palm_measured)
        self.assertTrue(self.jaw.friction_is_default)


class TheRegistryResolvesOrRefusesTests(unittest.TestCase):
    def test_the_short_name_the_corpora_carry_is_an_alias(self) -> None:
        self.assertEqual(load_gripper("2f85").model, "robotiq_2f85")

    def test_a_lookup_that_takes_model_names_only_refuses_the_short_name(self) -> None:
        """The short name 2f85 is how existing corpora stamp this hand, and the registry resolves it when
        asked to. A cell names its hand by the model name, so the build that reads robot.gripper.model
        looks it up with aliases=False, and a short name in a cell's config is refused there."""
        self.assertEqual(load_gripper("robotiq_2f85", aliases=False).model, "robotiq_2f85")
        with self.assertRaises(ConfigError) as ctx:
            load_gripper("2f85", aliases=False)
        self.assertIn("robotiq_2f85", str(ctx.exception))

    def test_the_shipped_hands_are_listed(self) -> None:
        self.assertIn("robotiq_2f85", available_grippers())

    def test_an_unknown_hand_is_refused_with_the_known_names(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            load_gripper("schunk_does_not_exist")
        self.assertIn("robotiq_2f85", str(ctx.exception))

    def test_a_name_that_is_not_registry_shaped_is_refused_before_a_file_is_read(self) -> None:
        for name in ("Robotiq_2F85", "robotiq-2f85", "../robotiq_2f85", ""):
            with self.subTest(name=name), self.assertRaises(ConfigError) as ctx:
                load_gripper(name)
            self.assertIn("not a registry name", str(ctx.exception))

    def test_a_tree_without_a_registry_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="willy_no_grippers_") as tmp:
            with self.assertRaises(ConfigError) as ctx:
                load_gripper("robotiq_2f85", data_dir=tmp)
        self.assertIn("no gripper registry", str(ctx.exception))


class ARegistryFileDescribesExactlyOneHandTests(unittest.TestCase):
    """Scratch registries built from the shipped file, each broken in exactly one way."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="willy_grippers_")
        self.root = Path(self._tmp) / "data"
        shutil.copytree(_DATA / "grippers", self.root / "grippers")
        self.shipped = (self.root / "grippers" / "robotiq_2f85.yaml").read_text(encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _write(self, name: str, text: str) -> None:
        (self.root / "grippers" / name).write_text(text, encoding="utf-8")

    def _refusal(self, name: str = "robotiq_2f85") -> str:
        with self.assertRaises(ConfigError) as ctx:
            load_gripper(name, data_dir=self.root)
        return str(ctx.exception)

    def _with_jaw(self, **numbers: object) -> str:
        text = self.shipped
        for key, value in numbers.items():
            lines = [line for line in text.splitlines() if line.strip().startswith(f"{key}:")]
            self.assertEqual(len(lines), 1, f"the shipped file stopped writing {key} once")
            indent = lines[0][: len(lines[0]) - len(lines[0].lstrip())]
            text = text.replace(lines[0], f"{indent}{key}: {value}")
        return text

    def test_a_file_whose_model_is_not_its_file_name_is_refused_for_that_reason(self) -> None:
        """Refused by the stem rule itself. Without that rule the two files collapse onto one model and
        the lookup of `some_other_hand` fails as an unknown name, which also names `some_other_hand`,
        so the message is held to the rule's own words."""
        self._write("some_other_hand.yaml", self.shipped)
        message = self._refusal("some_other_hand")
        self.assertIn("is named 'some_other_hand'", message)
        self.assertIn("describes the hand 'robotiq_2f85'", message)
        self.assertNotIn("Known hands", message)

    def test_an_alias_claimed_by_two_hands_is_refused(self) -> None:
        self.assertIn("model: robotiq_2f85", self.shipped)
        self._write("other_hand.yaml", self.shipped.replace("model: robotiq_2f85", "model: other_hand"))
        message = self._refusal("2f85")
        self.assertIn("2f85", message)
        self.assertIn("claimed by", message)

    def test_an_alias_that_is_another_hands_model_name_is_refused(self) -> None:
        other = (self.shipped.replace("model: robotiq_2f85", "model: other_hand")
                 .replace("aliases: [2f85]", "aliases: [robotiq_2f85]"))
        self.assertNotEqual(other, self.shipped.replace("model: robotiq_2f85", "model: other_hand"))
        self._write("other_hand.yaml", other)
        self.assertIn("model name of another hand", self._refusal())

    def test_a_hand_listed_as_its_own_alias_is_refused(self) -> None:
        self._write("robotiq_2f85.yaml", self.shipped.replace("aliases: [2f85]",
                                                              "aliases: [2f85, robotiq_2f85]"))
        self.assertIn("alias of itself", self._refusal())

    def test_pad_lengths_that_do_not_add_up_are_refused(self) -> None:
        self.assertIn("pad_behind_mm: 14.39", self.shipped)
        self._write("robotiq_2f85.yaml", self.shipped.replace("pad_behind_mm: 14.39", "pad_behind_mm: 20.0"))
        self.assertIn("pad", self._refusal())

    def test_a_malformed_file_is_refused_by_name(self) -> None:
        self._write("robotiq_2f85.yaml", "gripper: [never closed\n")
        self.assertIn("robotiq_2f85.yaml", self._refusal())

    def test_a_file_without_the_top_level_key_is_refused(self) -> None:
        self._write("robotiq_2f85.yaml", self.shipped.replace("gripper:", "hand:", 1))
        self.assertIn("top-level 'gripper:'", self._refusal())

    def test_widths_must_fit_strictly_inside_the_aperture(self) -> None:
        """The actuation schema refuses a minimum width equal to the maximum; the registry that is meant
        to feed it refuses the same."""
        for numbers in ({"min_width_mm": 85.0}, {"closed_width_mm": 85.0}):
            with self.subTest(**numbers):
                self._write("robotiq_2f85.yaml", self._with_jaw(**numbers))
                self.assertIn("aperture", self._refusal())

    def test_the_reach_and_the_pad_must_describe_one_jaw(self) -> None:
        cases = (
            ({"finger_length_mm": 30.0}, "finger_length_mm"),
            ({"pad_ahead_mm": 30.0, "pad_behind_mm": 8.0}, "pad_ahead_mm"),
            ({"pad_ahead_mm": 3.0, "pad_behind_mm": 35.0}, "pad_behind_mm"),
        )
        for numbers, named in cases:
            with self.subTest(**numbers):
                self._write("robotiq_2f85.yaml", self._with_jaw(**numbers))
                self.assertIn(named, self._refusal())

    def test_numbers_beyond_the_envelope_schemas_bounds_are_refused(self) -> None:
        """The registry is the source the envelope will be built from, so it cannot hold a value that
        schema would refuse."""
        for numbers in ({"finger_thickness_mm": 250.0}, {"finger_pad_overlap_mm": 150.0},
                        {"palm_width_mm": 600.0}):
            with self.subTest(**numbers):
                self._write("robotiq_2f85.yaml", self._with_jaw(**numbers))
                self._refusal()

    def test_a_number_that_is_not_finite_is_refused(self) -> None:
        """A friction of infinity would make every antipodal verdict pass once the labeller reads this
        file, and an infinite aperture passes every width relation."""
        for numbers in ({"aperture_mm": ".inf"}, {"friction_coefficient": ".inf"},
                        {"aperture_mm": ".nan"}, {"friction_coefficient": ".nan"}):
            with self.subTest(**numbers):
                self._write("robotiq_2f85.yaml", self._with_jaw(**numbers))
                self._refusal()

    def test_a_file_the_registry_cannot_read_as_a_hand_is_refused_on_every_platform(self) -> None:
        """A glob for `*.yaml` ignores `.yml` everywhere, matches `.YAML` only where the filesystem is
        case-insensitive, and reads a profile-style name as a hand named after a different stem. The
        registry names such a file instead of ignoring it on one platform and reading it on another.
        `other_hand.YAML` has a registry-shaped stem, so only the suffix rule can refuse it."""
        for name in ("robotiq_2f85.yml", "robotiq_2f85.sim.yaml", "Other_Hand.YAML", "other_hand.YAML"):
            with self.subTest(file=name):
                path = self.root / "grippers" / name
                path.write_text(self.shipped, encoding="utf-8")
                try:
                    message = self._refusal()
                    self.assertIn(name, message)
                    self.assertIn("<model>.yaml", message)
                finally:
                    path.unlink()

    def test_a_file_that_is_not_yaml_is_not_a_hand(self) -> None:
        self._write("README.md", "# the hands this cell may carry\n")
        self.assertEqual(load_gripper("robotiq_2f85", data_dir=self.root).model, "robotiq_2f85")


class TheCellNamesItsHandTests(unittest.TestCase):
    def test_a_tree_that_names_no_hand_loads_as_before(self) -> None:
        self.assertIsNone(RobotConfig().gripper.model)

    def test_a_registry_name_is_accepted(self) -> None:
        cfg = RobotConfig.model_validate({"gripper": {"model": "robotiq_2f85"}})
        self.assertEqual(cfg.gripper.model, "robotiq_2f85")

    def test_a_name_no_registry_file_defines_still_passes_the_schema(self) -> None:
        """Today's state, pinned so the step that reads the key flips it on purpose.

        The schema cannot see the registry: which files exist depends on the data directory a caller
        loads, and a schema is validated without one. So `robotiq_hande` passes here although no file
        describes it yet. The build that takes the hand from this key has to refuse such a name."""
        cfg = RobotConfig.model_validate({"gripper": {"model": "robotiq_hande"}})
        self.assertEqual(cfg.gripper.model, "robotiq_hande")
        self.assertNotIn("robotiq_hande", available_grippers())

    def test_the_short_name_still_passes_the_schema(self) -> None:
        """Today's state, pinned for the same reason: the schema checks the name's shape only, and the build
        refuses the short name through load_gripper(..., aliases=False)."""
        self.assertEqual(RobotConfig.model_validate({"gripper": {"model": "2f85"}}).gripper.model, "2f85")

    def test_a_name_that_cannot_be_a_registry_file_is_refused(self) -> None:
        """Refused by the NAME rule, not because the key is unknown.

        Before `model` existed every one of these was refused too, as an extra key, so a test that only
        asked for a refusal passed with the feature absent. The accepted name first, then a refusal
        whose error is about the value.
        """
        from pydantic import ValidationError

        RobotConfig.model_validate({"gripper": {"model": "robotiq_2f85"}})
        for bad in ("", "Robotiq_HandE", "robotiq-hande", "robotiq.hande", "../robotiq_2f85"):
            with self.subTest(model=bad):
                with self.assertRaises(ValidationError) as ctx:
                    RobotConfig.model_validate({"gripper": {"model": bad}})
                kinds = {error["type"] for error in ctx.exception.errors()}
                self.assertNotIn("extra_forbidden", kinds, "refused as an unknown key, not by the rule")


if __name__ == "__main__":
    unittest.main()
