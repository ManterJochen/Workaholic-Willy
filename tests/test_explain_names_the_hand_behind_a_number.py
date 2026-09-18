"""explain and decisions name the hand a derived number came from (customer chain lane C4c).

After lane C4b a cell naming a hand takes its widths and envelope from that hand's registry file. The explainer said
"no YAML sets this: the schema default is in force" about such a number, which is false twice: nothing defaulted,
and the file that decided it goes unnamed.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "config"
_SCHEMA_DEFAULT = "the schema default is in force"


def _loaded(case: unittest.TestCase, layer: str):
    from src.config.loader import reload_config
    from src.config.tree import ConfigTree

    root = Path(case.enterContext(tempfile.TemporaryDirectory())) / "data"
    shutil.copytree(_DATA, root)
    (root / "robot" / "robot.trial.yaml").write_text(layer, encoding="utf-8")
    reload_config()
    loaded = ConfigTree.from_directory(root=root, profile="trial").load()
    case.assertTrue(loaded.ok, loaded.error)
    return loaded


class ExplainNamesTheHandTests(unittest.TestCase):
    def test_a_derived_width_names_the_registry_file_and_field(self) -> None:
        explanation = _loaded(self, "robot:\n  gripper:\n    model: schunk_egu50\n").explain("robot.gripper.max_width_mm")
        self.assertEqual(explanation.value, 84.1)
        self.assertIn("config/grippers/schunk_egu50.yaml", explanation.set_in)
        self.assertIn("jaw.aperture_mm", explanation.set_in)
        self.assertNotIn(_SCHEMA_DEFAULT, explanation.render())

    def test_a_key_no_hand_supplies_still_says_the_schema_default(self) -> None:
        """⭐ THE CONTROL: a tree naming no hand, and a key outside the hand's thirteen, keep the old sentence."""
        no_hand = _loaded(self, "robot:\n  gripper:\n    model: null\n").explain("robot.gripper.max_width_mm")
        self.assertIn(_SCHEMA_DEFAULT, no_hand.render())
        self.assertEqual(no_hand.set_in, "")
        other = _loaded(self, "robot:\n  gripper:\n    model: schunk_egu50\n").explain(
            "robot.grasping.gripper_geometry.outer_margin_mm")
        self.assertIn(_SCHEMA_DEFAULT, other.render())

    def test_a_stated_number_names_its_layer_and_not_the_registry(self) -> None:
        explanation = _loaded(self, "robot:\n  gripper:\n    model: schunk_egu50\n    min_width_mm: 10.0\n").explain(
            "robot.gripper.min_width_mm")
        self.assertIn("robot.trial.yaml", explanation.set_in)
        self.assertNotIn("grippers", explanation.set_in)


class DecisionsNameTheHandTests(unittest.TestCase):
    def test_decisions_name_the_registry_for_a_derived_number(self) -> None:
        said = _loaded(self, "robot:\n  gripper:\n    model: schunk_egu50\n").decisions(section="robot.gripper")
        block = said[said.index("robot.gripper.max_width_mm"):]
        self.assertIn("config/grippers/schunk_egu50.yaml jaw.aperture_mm", block.splitlines()[1])

    def test_a_key_a_layer_writes_still_shows_that_layer(self) -> None:
        """⭐ THE CONTROL: robot.gripper.model is written by the trial layer and says so."""
        said = _loaded(self, "robot:\n  gripper:\n    model: schunk_egu50\n").decisions(section="robot.gripper")
        block = said[said.index("robot.gripper.model"):]
        self.assertIn("robot.trial.yaml", block.splitlines()[1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
