"""A named hand supplies its own widths at load, and a contradicting profile is refused (lane C4b, C4d).

The owner's decision of 2026-09-17 (finding 14 of the audit): a profile that named a customer hand and left
``grasping.gripper_geometry`` and the gripper widths unset planned grasps against a 2F-85's fingers and stroke. Since
2026-10-17 the envelope is no key at all: every reader takes it from the registry file (``gripper_model.jaw_envelope``),
and a tree that still writes the block is refused at load.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "config"


class _Tree:
    def __init__(self, case: unittest.TestCase, layer: str, *, registry: bool = True) -> None:
        self.root = Path(case.enterContext(tempfile.TemporaryDirectory())) / "data"
        shutil.copytree(_DATA, self.root)
        if not registry:
            shutil.rmtree(self.root / "grippers")
        (self.root / "robot" / "robot.trial.yaml").write_text(layer, encoding="utf-8")


def _load(tree: _Tree):
    from src.config.loader import load_config, reload_config

    reload_config()
    return load_config(tree.root, profile="trial").robot


class UnsetWidthsComeFromTheHandTests(unittest.TestCase):
    def test_unset_widths_come_from_the_named_hand_and_its_envelope_from_its_file(self) -> None:
        from src.config.grippers import load_gripper
        from src.config.hand_numbers import HAND_KEYS
        from src.robot.grasping.collision.gripper_model import jaw_envelope

        robot = _load(_Tree(self, "robot:\n  gripper:\n    model: schunk_egu50\n"))
        spec = load_gripper("schunk_egu50", aliases=False)
        for key, field in HAND_KEYS:
            with self.subTest(key=key):
                node = robot
                for part in key.split("."):
                    node = getattr(node, part)
                self.assertEqual(node, getattr(spec.jaw, field))
        envelope = jaw_envelope(robot)
        self.assertEqual((envelope.fingertip_depth_mm, envelope.palm_width_mm),
                         (spec.jaw.finger_ahead_mm, spec.jaw.palm_width_mm))

    def test_the_table_is_the_three_widths(self) -> None:
        """⭐ THE CONTROL: the loader fills the actuation widths and nothing else."""
        from src.config.hand_numbers import HAND_KEYS

        fields = {field for _, field in HAND_KEYS}
        self.assertEqual(fields, {"aperture_mm", "min_width_mm", "closed_width_mm"})

    def test_the_robot_section_door_derives_the_same_numbers(self) -> None:
        from src.config.loader import load_robot_section

        tree = _Tree(self, "robot:\n  gripper:\n    model: schunk_egu50\n")
        section = load_robot_section(tree.root, profile="trial")
        self.assertEqual(section.gripper.closed_width_mm, 4.1)
        self.assertEqual(section, _load(tree))

    def test_a_tree_naming_no_hand_keeps_the_schema_defaults(self) -> None:
        """⭐ THE CONTROL: no hand named, nothing derived, the base tree loads as it did."""
        from src.config.schema.robot import GripperConfig

        robot = _load(_Tree(self, "robot:\n  gripper:\n    model: null\n"))
        self.assertEqual(robot.gripper.max_width_mm, GripperConfig().max_width_mm)


class AContradictionIsRefusedTests(unittest.TestCase):
    def test_a_stated_number_that_contradicts_the_named_hand_is_refused_naming_both(self) -> None:
        from src.config.loader import ConfigError

        layer = "robot:\n  gripper:\n    model: schunk_egu50\n    max_width_mm: 85.0\n    closed_width_mm: 0.0\n"
        with self.assertRaises(ConfigError) as caught:
            _load(_Tree(self, layer))
        said = str(caught.exception)
        for part in ("robot.gripper.max_width_mm", "85.0", "84.1", "robot.gripper.closed_width_mm", "4.1",
                     "grippers/schunk_egu50.yaml"):
            self.assertIn(part, said)

    def test_a_restated_equal_number_and_a_policy_inside_its_range_are_admitted(self) -> None:
        layer = "robot:\n  gripper:\n    model: schunk_egu50\n    max_width_mm: 84.1\n    min_width_mm: 10.0\n"
        robot = _load(_Tree(self, layer))
        self.assertEqual((robot.gripper.max_width_mm, robot.gripper.min_width_mm), (84.1, 10.0))

    def test_a_tree_holding_no_registry_takes_the_numbers_from_the_repository_registry(self) -> None:
        """The owner's one registry authority: a generated data_example tree has no grippers/ and still loads a hand."""
        robot = _load(_Tree(self, "robot:\n  gripper:\n    model: schunk_egu50\n", registry=False))
        self.assertEqual(robot.gripper.closed_width_mm, 4.1)

    def test_the_numbers_come_from_the_repository_registry_and_not_the_tree_s_copy(self) -> None:
        """⭐ THE CONTROL on the authority: a tree whose copy says otherwise does not move a loaded number.

        Refusing that tree is the validator's and the desk's (lane C1f, C1g), with both files named.
        """
        tree = _Tree(self, "robot:\n  gripper:\n    model: schunk_egu50\n")
        path = tree.root / "grippers" / "schunk_egu50.yaml"
        text = path.read_text(encoding="utf-8")
        self.assertIn("closed_width_mm: 4.1", text)
        path.write_text(text.replace("closed_width_mm: 4.1", "closed_width_mm: 9.9"), encoding="utf-8")
        self.assertEqual(_load(tree).gripper.closed_width_mm, 4.1)

    def test_the_refusal_names_the_repository_registry_file(self) -> None:
        from src.config.loader import ConfigError

        with self.assertRaises(ConfigError) as caught:
            _load(_Tree(self, "robot:\n  gripper:\n    model: schunk_egu50\n    closed_width_mm: 0.0\n"))
        self.assertIn("config/grippers/schunk_egu50.yaml", str(caught.exception))

    def test_a_tree_still_writing_the_envelope_is_refused_naming_the_registry_file(self) -> None:
        from src.config.loader import ConfigError

        layer = "robot:\n  gripper:\n    model: schunk_egu50\n  grasping:\n    gripper_geometry:\n      kind: suction\n"
        with self.assertRaises(ConfigError) as caught:
            _load(_Tree(self, layer))
        for part in ("robot.grasping.gripper_geometry", "config/grippers", "delete it"):
            self.assertIn(part, str(caught.exception))


class NoShippedLayerRestatesTheRegistryTests(unittest.TestCase):
    def test_no_shipped_robot_layer_states_a_number_the_registry_holds(self) -> None:
        """C4d. One description per hand: a shipped layer that names a hand restates none of its numbers."""
        from src.config.hand_numbers import HAND_KEYS

        offenders = []
        for path in sorted((_DATA / "robot").glob("robot*.yaml")):
            robot = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("robot") or {}
            for key, _ in HAND_KEYS:
                node = robot
                for part in key.split("."):
                    node = node.get(part) if isinstance(node, dict) else None
                if node is not None:
                    offenders.append(f"{path.name}: robot.{key}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
