"""Referring expressions — the one artefact here whose correctness is a language question.

Two things are being protected. That an expression is UNIQUE, because a prompt matching two objects is
not ground truth; and that the German is grammatical, because a grounding benchmark written in broken
German measures a model's tolerance for broken German instead of its grounding.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.prompts.colours import name_colour
from datagen.prompts.expressions import (
    KIND_WORDS,
    ObjectFacts,
    attribute_prompts,
    scene_facts,
    spatial_prompts,
)


def _fact(index: int, kind: str, rgb: tuple[float, float, float], longest: float = 50.0,
          x: float = 0.0) -> ObjectFacts:
    return ObjectFacts(index, kind, name_colour(rgb), longest, (x, 0.0, 0.0))


_RED, _BLUE, _MUD = (0.85, 0.08, 0.08), (0.08, 0.12, 0.85), (0.55, 0.50, 0.42)


class ColourNamingTests(unittest.TestCase):
    def test_a_clear_colour_is_named_in_both_languages(self) -> None:
        colour = name_colour(_RED)
        assert colour is not None
        self.assertEqual((colour.en, colour.de), ("red", "rot"))

    def test_a_boundary_hue_is_refused_rather_than_guessed(self) -> None:
        """Hue 45 deg sits between orange and yellow; two people would disagree, so neither is used."""
        import colorsys

        between = colorsys.hsv_to_rgb(45.0 / 360.0, 0.85, 0.85)
        self.assertIsNone(name_colour(between))

    def test_a_washed_out_colour_is_refused(self) -> None:
        self.assertIsNone(name_colour(_MUD))

    def test_black_and_white_are_named_without_a_hue(self) -> None:
        for rgb, expected in (((0.03, 0.03, 0.04), "black"), ((0.95, 0.95, 0.93), "white")):
            colour = name_colour(rgb)
            assert colour is not None, rgb
            self.assertEqual(colour.en, expected)


class AttributeTests(unittest.TestCase):
    def test_a_lone_shape_needs_no_colour(self) -> None:
        """The shortest true description is the one a person says; over-specifying makes it easier."""
        prompts = attribute_prompts("s", [_fact(0, "can", _RED), _fact(1, "box", _BLUE)])
        english = {p.target: p.text for p in prompts if p.language == "en"}
        self.assertEqual(english[0], "the can")
        self.assertEqual(english[1], "the box")

    def test_colour_separates_two_of_a_kind(self) -> None:
        prompts = attribute_prompts("s", [_fact(0, "can", _RED), _fact(1, "can", _BLUE)])
        english = {p.target: p.text for p in prompts if p.language == "en"}
        self.assertEqual(english, {0: "the red can", 1: "the blue can"})

    def test_two_identical_objects_get_NO_attribute_prompt(self) -> None:
        """The invariant. Emitting 'the red can' for either of two red cans is not ground truth."""
        prompts = attribute_prompts("s", [_fact(0, "can", _RED), _fact(1, "can", _RED)])
        self.assertEqual(prompts, [])

    def test_an_unnameable_colour_does_not_become_a_prompt(self) -> None:
        prompts = attribute_prompts("s", [_fact(0, "can", _MUD), _fact(1, "can", _RED)])
        targets = {p.target for p in prompts}
        self.assertNotIn(0, targets, "an object whose colour has no defensible name cannot use one")

    def test_size_separates_two_of_a_kind_when_colour_cannot(self) -> None:
        prompts = attribute_prompts(
            "s", [_fact(0, "can", _MUD, longest=100.0), _fact(1, "can", _MUD, longest=40.0)])
        english = {p.target: p.text for p in prompts if p.language == "en"}
        self.assertEqual(english, {0: "the large can", 1: "the small can"})

    def test_every_prompt_comes_in_both_languages(self) -> None:
        prompts = attribute_prompts("s", [_fact(0, "can", _RED), _fact(1, "box", _BLUE)])
        self.assertEqual(sum(1 for p in prompts if p.language == "de"),
                         sum(1 for p in prompts if p.language == "en"))


class GermanGrammarTests(unittest.TestCase):
    def test_the_article_matches_the_noun(self) -> None:
        """Two of each kind, so colour is needed and the adjective+article have to agree."""
        prompts = attribute_prompts("s", [
            _fact(0, "cup", _RED), _fact(1, "cup", _BLUE),
            _fact(2, "can", _RED), _fact(3, "can", _BLUE),
        ])
        german = {p.target: p.text for p in prompts if p.language == "de"}
        self.assertEqual(german[0], "der rote Becher", "Becher is masculine")
        self.assertEqual(german[3], "die blaue Dose", "Dose is feminine")

    def test_von_takes_the_dative(self) -> None:
        """The defect this test was written for: the first draft said 'links von die rote Box'."""
        facts = [_fact(0, "can", _MUD, x=0.0), _fact(1, "can", _MUD, x=200.0),
                 _fact(2, "box", _RED, x=100.0)]
        pixels = np.asarray([[10.0, 100.0], [400.0, 100.0], [200.0, 100.0]])
        prompts = spatial_prompts("s", facts, "oblique_left", pixels)
        german = [p.text for p in prompts if p.language == "de"]
        self.assertTrue(german, "the two identical cans should need a spatial expression")
        for text in german:
            self.assertIn("von der roten Box", text)
            self.assertNotIn("von die", text)

    def test_every_kind_has_a_german_word_and_an_article(self) -> None:
        from datagen.prompts.expressions import _GERMAN_ARTICLE  # noqa: PLC0415

        for kind, (english, german) in KIND_WORDS.items():
            self.assertTrue(english and german, kind)
            self.assertIn(german, _GERMAN_ARTICLE, f"{kind}: no article for {german!r}")


class InstructionTests(unittest.TestCase):
    """The instruction class: a wrapper, so it must inherit uniqueness and fix only the grammar."""

    def _facts(self):  # noqa: ANN202
        return [_fact(0, "cup", _RED), _fact(1, "can", _BLUE)]

    def test_the_german_target_is_accusative(self) -> None:
        """'Nimm der blaue Becher auf' is wrong; the imperative takes the accusative."""
        from datagen.prompts.expressions import instruction_prompts

        facts = [_fact(0, "cup", _RED), _fact(1, "cup", _BLUE)]
        base = attribute_prompts("s", facts)
        german = [p.text for p in instruction_prompts(base, facts) if p.language == "de"]
        self.assertTrue(german)
        for text in german:
            self.assertNotIn("der rote Becher", text, "nominative leaked into an imperative")
            self.assertIn("den ", text, "masculine accusative is 'den'")

    def test_a_separable_verb_puts_its_particle_last(self) -> None:
        from datagen.prompts.expressions import instruction_prompts

        facts = self._facts()
        texts = [p.text for p in instruction_prompts(attribute_prompts("s", facts), facts)
                 if p.language == "de" and p.text.startswith("Nimm")]
        for text in texts:
            self.assertTrue(text.endswith(" auf"), f"separable particle misplaced: {text!r}")

    def test_the_verb_is_reproducible(self) -> None:
        """'Randomly mixed' still has to mean the same thing on the second run."""
        from datagen.prompts.expressions import instruction_prompts

        facts = self._facts()
        base = attribute_prompts("s", facts)
        first = [p.text for p in instruction_prompts(base, facts)]
        second = [p.text for p in instruction_prompts(base, facts)]
        self.assertEqual(first, second)

    def test_it_never_invents_a_target(self) -> None:
        from datagen.prompts.expressions import instruction_prompts

        facts = self._facts()
        base = attribute_prompts("s", facts)
        wrapped = instruction_prompts(base, facts)
        self.assertEqual({p.target for p in wrapped}, {p.target for p in base})


class SpatialTests(unittest.TestCase):
    def test_the_relation_must_single_the_target_out(self) -> None:
        """Three identical cans left of one box: 'the can left of the box' matches two, so nothing."""
        facts = [_fact(0, "can", _MUD), _fact(1, "can", _MUD), _fact(2, "box", _RED)]
        pixels = np.asarray([[10.0, 100.0], [50.0, 100.0], [300.0, 100.0]])
        self.assertEqual(spatial_prompts("s", facts, "v", pixels), [])

    def test_a_spatial_prompt_records_the_view_it_is_valid_for(self) -> None:
        facts = [_fact(0, "can", _MUD), _fact(1, "can", _MUD), _fact(2, "box", _RED)]
        pixels = np.asarray([[10.0, 100.0], [400.0, 100.0], [200.0, 100.0]])
        for prompt in spatial_prompts("s", facts, "wrist", pixels):
            self.assertEqual(prompt.view, "wrist", "left-of is image-space; it belongs to one camera")


class SceneFactsTests(unittest.TestCase):
    def test_a_dropped_object_never_becomes_a_target(self) -> None:
        payload = {
            "dropped_objects": [1],
            "spec": {"objects": [
                {"asset_id": "proc_00000_can", "color_rgb": list(_RED), "position_mm": [0, 0, 0]},
                {"asset_id": "proc_00001_box", "color_rgb": list(_BLUE), "position_mm": [0, 0, 0]},
            ]},
            "settled_poses_mm_xyzw": {"0": [[1.0, 2.0, 3.0], [0, 0, 0, 1]]},
        }
        facts = scene_facts(payload)
        self.assertEqual([fact.index for fact in facts], [0])
        self.assertEqual(facts[0].position_mm, (1.0, 2.0, 3.0), "the SETTLED pose, not the planned one")


if __name__ == "__main__":
    unittest.main()
