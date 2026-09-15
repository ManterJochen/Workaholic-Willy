"""One resolver answers which numbers a hand's name means, for the network's conditioning vector.

Two sources describe jaws, and lane (i) makes them one question. The gripper registry
(``config/grippers/``) describes the hands this repository owns, one file each, and ``JAW_GEOMETRY``
(``deep/net/gripper.py``) the three procedural jaws a corpus was labelled with so the conditioning varies, which
are not grippers anybody owns. ``resolve_hand`` asks the registry first, aliases included, so the ``2f85`` stamp
every existing corpus and artifact carries is the Robotiq 2F-85's file, and answers from ``JAW_GEOMETRY`` only for
a name the registry does not know. A name both sources know with different numbers is refused, and so is a name
neither knows.

Byte-identical by construction for every stamp in use: the 2F-85 file is held equal to ``JAW_GEOMETRY["2f85"]``
(tests/test_gripper_registry.py), and the procedural jaws are read from ``JAW_GEOMETRY`` itself.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.config.grippers import load_gripper
from src.robot.grasping.deep.corpus.sample import SampleSpec
from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY, gripper_vector, swept_volume_vector

_DATA = Path(__file__).resolve().parents[1] / "config"
_PROCEDURAL = ("narrow_55", "wide_140", "slim_pad")


def _registry_numbers(model: str) -> dict[str, float]:
    jaw = load_gripper(model, aliases=False).jaw
    return {
        "aperture_mm": jaw.aperture_mm, "min_width_mm": jaw.min_width_mm,
        "finger_ahead_mm": jaw.finger_ahead_mm, "finger_behind_mm": jaw.finger_behind_mm,
        "finger_thickness_mm": jaw.finger_thickness_mm, "finger_width_mm": jaw.finger_width_mm,
        "pad_span_mm": jaw.pad_length_mm, "friction_coefficient": jaw.friction_coefficient,
    }


class TheResolverTests(unittest.TestCase):
    def test_the_2f85_stamp_and_its_model_name_give_the_trained_vector(self) -> None:
        from src.robot.grasping.deep.hands import hand_vector

        trained = gripper_vector("2f85")
        self.assertTrue(torch.equal(hand_vector("2f85"), trained))
        self.assertTrue(torch.equal(hand_vector("robotiq_2f85"), trained))

    def test_a_stamp_and_its_model_name_are_one_hand(self) -> None:
        from src.robot.grasping.deep.hands import resolve_hand

        self.assertEqual(resolve_hand("2f85").model, "robotiq_2f85")
        self.assertEqual(resolve_hand("robotiq_2f85").model, "robotiq_2f85")

    def test_a_registry_hand_resolves_without_a_jaw_geometry_entry(self) -> None:
        from src.robot.grasping.deep.hands import resolve_hand

        hand = resolve_hand("robotiq_hande")
        self.assertEqual(hand.model, "robotiq_hande")
        self.assertEqual(hand.numbers, _registry_numbers("robotiq_hande"))
        with self.assertRaises(ValueError):
            gripper_vector("robotiq_hande")  # the net's own lookup still knows the procedural names only

    def test_a_procedural_jaw_still_resolves_from_jaw_geometry(self) -> None:
        from src.robot.grasping.deep.hands import hand_vector, resolve_hand

        for name in _PROCEDURAL:
            with self.subTest(jaw=name):
                hand = resolve_hand(name)
                self.assertEqual(hand.model, name)
                self.assertEqual(hand.numbers, JAW_GEOMETRY[name])
                self.assertTrue(torch.equal(hand_vector(name), gripper_vector(name)))

    def test_a_name_neither_source_knows_is_refused_naming_both(self) -> None:
        from src.robot.grasping.deep.hands import resolve_hand

        with self.assertRaises(ValueError) as caught:
            resolve_hand("a_hand_we_never_labelled")
        message = str(caught.exception)
        self.assertIn("unknown gripper stamp", message)
        self.assertIn("robotiq_hande", message)
        self.assertIn("narrow_55", message)

    def test_a_registry_hand_that_contradicts_its_jaw_geometry_entry_is_refused(self) -> None:
        from src.robot.grasping.deep.hands import resolve_hand

        with TemporaryDirectory(prefix="willy_hands_") as tmp:
            root = Path(tmp) / "data"
            shutil.copytree(_DATA / "grippers", root / "grippers")
            path = root / "grippers" / "robotiq_2f85.yaml"
            text = path.read_text(encoding="utf-8")
            self.assertIn("aperture_mm: 85.0", text)
            path.write_text(text.replace("aperture_mm: 85.0", "aperture_mm: 86.0"), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                resolve_hand("2f85", data_dir=root)
        message = str(caught.exception)
        self.assertIn("2f85", message)
        self.assertIn("aperture_mm", message)


class TheTrainerAsksTheResolverTests(unittest.TestCase):
    def test_a_cloud_stamped_with_a_registry_hand_gets_that_hands_vector(self) -> None:
        """Today the first batch refuses the stamp, because the trainer's lookup knows JAW_GEOMETRY names only."""
        from src.robot.grasping.deep.train import trainer
        from tests.test_deep_set_loop import _corpus

        with TemporaryDirectory() as tmp:
            index = trainer.corpus_index(_corpus(Path(tmp), scenes=2, gripper="robotiq_hande"))
            _, rows = trainer._batch_samples(  # noqa: SLF001
                index, [0], SampleSpec(grasp_set=True, points=256), np.random.default_rng(0),
            )
        self.assertTrue(torch.equal(rows[0], swept_volume_vector(**_registry_numbers("robotiq_hande"))))


if __name__ == "__main__":
    unittest.main()
