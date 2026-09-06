"""The gripper description the generator is conditioned on.

⚠ TWO OF THESE TESTS EXIST BECAUSE OF A CLAIM THE MODULE MAKES ABOUT ITSELF, not because of a shape.
It says eight of the twelve published numbers are constant for a parallel jaw, and it says the whole
vector is constant while we own one gripper. Both are reasons to distrust a future result, so both
are pinned here: if a change ever makes them false, that is good news and the test should be updated
with the measurement that made it false.

The third reason is drift. The numbers come from the 2F-85's collision shapes, measured 2026-08-14,
and they are spelled in THREE places: the backend collision model, the labeller's own reference, and
whatever a caller passes here. This file holds the first two together, which is the same pattern
`test_approach_granularity` uses for the feature-channel constants.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel
from src.robot.grasping.deep.net.gripper import (
    GRIPPER_LENGTH_SCALE_MM,
    GRIPPER_VECTOR_DIM,
    JAW_GEOMETRY,
    GripperEncoder,
    gripper_vector,
    swept_volume_vector,
)

#: The 2F-85, as the LABELLER models it. The corpus was labelled with these numbers, so they are the
#: authority for a training input even where the backend's collision model parameterises differently.
_JAW = {
    "aperture_mm": 85.0,
    "min_width_mm": 5.0,
    "finger_ahead_mm": 28.72,
    "finger_behind_mm": 33.37,
    "finger_thickness_mm": 31.35,
    "finger_width_mm": 27.0,
    "pad_span_mm": 38.0,
    "friction_coefficient": 0.5,
}


class VectorTests(unittest.TestCase):

    def test_the_shape_is_the_published_twelve_plus_our_two(self) -> None:
        self.assertEqual(swept_volume_vector(**_JAW).shape, (GRIPPER_VECTOR_DIM,))

    def test_every_length_is_scaled_into_order_one(self) -> None:
        """A jaw is order 100 mm and the feature channels beside it are order 1. This repository
        measured a 1:71 imbalance from exactly that mismatch and the learned weights did not
        compensate."""
        vector = swept_volume_vector(**_JAW)
        self.assertLess(float(vector.abs().max()), 2.0)
        self.assertGreater(float(vector.abs().max()), 0.1)

    def test_the_approach_centre_is_not_zero_because_a_real_jaw_is_not_symmetric(self) -> None:
        """MEASURED 28.72 mm ahead against 33.37 mm behind. A representation that assumed symmetry
        would place the whole finger 2.3 mm off, in the direction of the table."""
        vector = swept_volume_vector(**_JAW)
        expected = (28.72 - 33.37) / 2.0 / GRIPPER_LENGTH_SCALE_MM
        self.assertAlmostEqual(float(vector[5]), expected, places=6)
        self.assertLess(float(vector[5]), 0.0)

    def test_only_the_closing_extent_differs_between_the_two_states(self) -> None:
        """⚠ THE DEGENERACY THE MODULE WARNS ABOUT, pinned. Of the twelve published numbers, ours
        vary in exactly one. A cross-gripper result read off this vector has to be read knowing that.
        """
        vector = swept_volume_vector(**_JAW)
        wide, half = vector[:6], vector[6:12]
        self.assertNotAlmostEqual(float(wide[0]), float(half[0]), places=4)
        for index in range(1, 6):
            self.assertAlmostEqual(float(wide[index]), float(half[index]), places=9)

    def test_the_two_symmetric_centres_are_exactly_zero(self) -> None:
        vector = swept_volume_vector(**_JAW)
        for index in (3, 4, 9, 10):
            self.assertEqual(float(vector[index]), 0.0)

    def test_the_closing_extent_accounts_for_the_finger_bodies(self) -> None:
        """The swept region is not the opening: each finger occupies its own thickness outside the
        pad face, and a box that stopped at the pads would let the housing pass through an obstacle.
        """
        vector = swept_volume_vector(**_JAW)
        expected = (85.0 + 2.0 * 31.35) / GRIPPER_LENGTH_SCALE_MM
        self.assertAlmostEqual(float(vector[0]), expected, places=6)

    def test_a_wider_jaw_produces_a_different_vector(self) -> None:
        """The liveness precondition. If this were false, no amount of extra labelling could make the
        conditioning carry a gradient."""
        base = swept_volume_vector(**_JAW)
        wider = swept_volume_vector(**{**_JAW, "aperture_mm": 140.0})
        self.assertGreater(float((base - wider).abs().max()), 0.1)

    def test_refusals(self) -> None:
        with self.assertRaises(ValueError):
            swept_volume_vector(**{**_JAW, "aperture_mm": 0.0})
        with self.assertRaises(ValueError):
            swept_volume_vector(**{**_JAW, "min_width_mm": 90.0})   # wider than the aperture


class DriftTests(unittest.TestCase):

    def test_the_jaw_numbers_match_the_backend_collision_model(self) -> None:
        """Held together rather than assumed equal. These are the same measured shapes, and a change
        to one of them that did not reach the other would train the net on a gripper we do not have.
        """
        model = ParallelJawGripperModel()
        self.assertAlmostEqual(_JAW["finger_ahead_mm"], model.fingertip_depth_mm, places=6)
        self.assertAlmostEqual(_JAW["finger_thickness_mm"], model.finger_thickness_mm, places=6)
        self.assertAlmostEqual(_JAW["finger_width_mm"], model.finger_width_mm, places=6)
        self.assertAlmostEqual(_JAW["pad_span_mm"], model.pad_length_mm, places=6)

    def test_the_jaw_numbers_match_the_labeller(self) -> None:
        """The corpus was labelled with the labeller's model, so it is the authority for a training
        input. Imported here rather than in the module because `datagen` sits ABOVE `backend` and the
        dependency may never run that way outside a test."""
        from datagen.grasps.verdict import JawModel  # noqa: PLC0415

        jaw = JawModel()
        self.assertAlmostEqual(_JAW["aperture_mm"], jaw.aperture_mm, places=6)
        self.assertAlmostEqual(_JAW["min_width_mm"], jaw.min_width_mm, places=6)
        self.assertAlmostEqual(_JAW["finger_ahead_mm"], jaw.finger_ahead_mm, places=6)
        self.assertAlmostEqual(_JAW["finger_behind_mm"], jaw.finger_behind_mm, places=6)
        self.assertAlmostEqual(_JAW["finger_thickness_mm"], jaw.finger_thickness_mm, places=6)
        self.assertAlmostEqual(_JAW["finger_width_mm"], jaw.finger_width_mm, places=6)
        self.assertAlmostEqual(_JAW["friction_coefficient"], jaw.friction_coefficient, places=6)
        self.assertAlmostEqual(_JAW["pad_span_mm"], jaw.pad_ahead_mm + jaw.pad_behind_mm, places=6)


class EncoderTests(unittest.TestCase):

    def test_it_maps_a_batch_to_the_asked_width(self) -> None:
        encoder = GripperEncoder(width=32)
        out = encoder(swept_volume_vector(**_JAW).expand(4, GRIPPER_VECTOR_DIM))
        self.assertEqual(out.shape, (4, 32))

    def test_a_wrong_length_is_refused_rather_than_broadcast(self) -> None:
        encoder = GripperEncoder(width=16)
        with self.assertRaises(ValueError):
            encoder(torch.zeros(4, GRIPPER_VECTOR_DIM - 1))

    def test_two_different_grippers_do_not_collapse_to_one_embedding(self) -> None:
        """A precondition for the seam being able to work at all, separate from whether it DOES."""
        encoder = GripperEncoder(width=32).eval()
        with torch.no_grad():
            narrow = encoder(swept_volume_vector(**_JAW).unsqueeze(0))
            wide = encoder(swept_volume_vector(**{**_JAW, "aperture_mm": 140.0}).unsqueeze(0))
        self.assertGreater(float((narrow - wide).abs().max()), 1e-4)

    def test_a_constant_input_carries_no_gradient_into_the_vector(self) -> None:
        """⛔ THE INERTNESS THIS DESIGN IS EXPOSED TO, demonstrated rather than described.

        With one gripper every sample sees the same fourteen numbers. The encoder still trains, but
        nothing distinguishes the samples, so the embedding is a learned constant and the seam is
        wiring without a capability. This test does not fail when that is true: it PINS that it is
        true, so a future differential test has something to be different from.
        """
        encoder = GripperEncoder(width=8)
        vector = swept_volume_vector(**_JAW).expand(6, GRIPPER_VECTOR_DIM)
        embedded = encoder(vector)
        spread = embedded.std(dim=0).max()
        self.assertLess(float(spread), 1e-6,
                        "identical gripper inputs must produce identical embeddings")


class StampTests(unittest.TestCase):
    """The name every cloud is stamped with has to resolve to the jaw the LABELS came from."""

    def test_the_registry_matches_the_labeller_for_every_jaw(self) -> None:
        """⛔ SPELLED TWICE, HELD TOGETHER HERE. `net/` may not import the labelling layer, so these
        numbers are a copy, and a copy that drifted would train the head on a hand that does not
        exist. Imported inside the test because `datagen` sits ABOVE `backend`."""
        from datagen.grasps.verdict import PROCEDURAL_JAWS, JawModel

        known = {"2f85": JawModel(), **PROCEDURAL_JAWS}
        self.assertEqual(set(JAW_GEOMETRY), set(known),
                         "the registry and the labeller know different jaws")
        for name, jaw in known.items():
            with self.subTest(name):
                ours = JAW_GEOMETRY[name]
                self.assertAlmostEqual(ours["aperture_mm"], jaw.aperture_mm, places=6)
                self.assertAlmostEqual(ours["min_width_mm"], jaw.min_width_mm, places=6)
                self.assertAlmostEqual(ours["finger_ahead_mm"], jaw.finger_ahead_mm, places=6)
                self.assertAlmostEqual(ours["finger_behind_mm"], jaw.finger_behind_mm, places=6)
                self.assertAlmostEqual(ours["finger_thickness_mm"], jaw.finger_thickness_mm,
                                       places=6)
                self.assertAlmostEqual(ours["finger_width_mm"], jaw.finger_width_mm, places=6)
                self.assertAlmostEqual(ours["pad_span_mm"], jaw.pad_ahead_mm + jaw.pad_behind_mm,
                                       places=6)
                self.assertAlmostEqual(ours["friction_coefficient"], jaw.friction_coefficient,
                                       places=6)

    def test_an_unknown_stamp_raises_rather_than_defaulting(self) -> None:
        """⛔ THE REFUSAL IS THE POINT. A corpus labelled for a hand this module does not know would
        otherwise be trained as if it were the 2F-85, and a WRONG conditioning input is worse than a
        constant one: constant teaches nothing, wrong teaches a relationship that is not there."""
        with self.assertRaises(ValueError) as raised:
            gripper_vector("some_gripper_we_never_labelled")
        self.assertIn("Refusing to guess", str(raised.exception))

    def test_every_registered_jaw_produces_a_distinct_vector(self) -> None:
        vectors = {name: gripper_vector(name) for name in JAW_GEOMETRY}
        names = sorted(vectors)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                with self.subTest(f"{first} vs {second}"):
                    self.assertGreater(float((vectors[first] - vectors[second]).abs().max()), 1e-6)

    def test_the_pad_only_jaw_differs_from_the_shipped_one_in_exactly_one_number(self) -> None:
        """⭐ `slim_pad` has the 2F-85's opening and half its contact patch. That is what lets a
        differential test tell the PAD channels apart from the width ones, and it is only true if
        exactly one of the fourteen moves."""
        base, slim = gripper_vector("2f85"), gripper_vector("slim_pad")
        differing = int((base - slim).abs().gt(1e-9).sum())
        self.assertEqual(differing, 1)
        self.assertNotAlmostEqual(float(base[12]), float(slim[12]), places=6)


if __name__ == "__main__":
    unittest.main()
