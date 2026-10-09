"""A part of another colour is no target: the colour a phrase names, read against the part's own pixels (F1).

Asked for "each separate grey cube", the detector on the owner's cell boxed the green and the orange parts as well and
called every box by the prompt's words (2026-10-07 and 08), and the task picked them. ``src/robot/perception/
colour_check.py`` reads the one colour word of the phrase's head, English or German, and judges each part's mask on it
in CIELAB; only where the pixels cannot tell is the VLM asked for one colour word, and no answer refuses the part.

These pin the rule with no model. The patches are painted at the Lab values measured on the cell's own home views
(grey cubes L* 62-69 and C* 1-4, green parts C* 30-32 at hue 151-153 degrees, white cylinders L* 92-100, orange C* 80 at
hue 55 degrees, the black mat L* 16 and C* 4), each with a 4 px rim of the mat inside its mask, as a mask's edge bleeds
the mat in.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.perception.colour_check import (
    UNNAMED,
    Colour,
    ColourVerdict,
    colour_named,
    colour_of_word,
    judge_colour,
    refused_label,
)


def _bgr_of_lab(lightness: float, chroma: float, hue_deg: float) -> tuple[int, int, int]:
    """The 8-bit BGR colour of a CIELAB colour, through OpenCV's own conversion (the one the check reads with)."""
    import cv2

    a, b = chroma * math.cos(math.radians(hue_deg)), chroma * math.sin(math.radians(hue_deg))
    lab = np.array([[[lightness * 255.0 / 100.0, a + 128.0, b + 128.0]]], dtype=np.float32)
    bgr = cv2.cvtColor(np.clip(np.round(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])


GREY = _bgr_of_lab(66.0, 2.0, 90.0)
WHITE = _bgr_of_lab(95.0, 2.5, 90.0)
GREEN = _bgr_of_lab(60.0, 31.0, 152.0)
#: L* 62, so its red channel (242) stays under the camera's clip: at L* 65 it reads 252, a pixel the check leaves out
#: since 2026-10-09 (``tests/test_the_colour_check_leaves_clipped_pixels_out.py`` paints the clipped ones).
ORANGE = _bgr_of_lab(62.0, 80.0, 55.0)
MAT = _bgr_of_lab(16.0, 4.0, 90.0)
#: The red part ``tests/test_pick_prompt.py`` paints.
RED = (40, 40, 200)


def _patch(colour: tuple[int, int, int], *, size: int = 30, rim: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A ``size`` px square part on the mat, its mask the whole square and its outer ``rim`` px the mat's colour."""
    image = np.empty((80, 80, 3), dtype=np.uint8)
    image[...] = MAT
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:20 + size, 20:20 + size] = True
    image[20 + rim:20 + size - rim, 20 + rim:20 + size - rim] = colour
    return image, mask


def _verdict(colour: tuple[int, int, int], wanted: Colour, **patch: int) -> ColourVerdict:
    image, mask = _patch(colour, **patch)
    return judge_colour(image, mask, wanted)


class TheColourWordsTests(unittest.TestCase):
    def test_grey_is_named_in_english_and_german_spelled_either_way(self) -> None:
        for phrase in ("grey cube", "gray cube", "grauer Würfel auf der schwarzen Matte.", "silver bolt",
                       "each separate grey cube"):
            with self.subTest(phrase=phrase):
                self.assertIs(Colour.GREY, colour_named(phrase))

    def test_a_phrase_with_no_colour_or_two_names_none(self) -> None:
        for phrase in ("cube", "red and white part", "each separate object", ""):
            with self.subTest(phrase=phrase):
                self.assertIsNone(colour_named(phrase))

    def test_where_a_part_lies_names_no_colour_of_it(self) -> None:
        self.assertIs(Colour.GREY, colour_named("grey cube on the black mat"))
        self.assertIs(Colour.RED, colour_named("the red cube next to the blue bin"))
        self.assertIs(Colour.ORANGE, colour_named("an orange cube"), "the article an is no place word")

    def test_german_endings_are_read(self) -> None:
        for phrase, colour in (("roten Teile", Colour.RED), ("weißen Zylinder", Colour.WHITE),
                               ("grünen Teile", Colour.GREEN), ("orangen Würfel", Colour.ORANGE),
                               ("silberne Schraube", Colour.GREY), ("blaue Kiste", Colour.BLUE),
                               ("gruene Teile", Colour.GREEN)):
            with self.subTest(phrase=phrase):
                self.assertIs(colour, colour_named(phrase))

    def test_a_vlm_answer_is_read_through_the_same_words(self) -> None:
        for answer, colour in (("Grey.", Colour.GREY), ("silver", Colour.GREY), ("light grey", Colour.GREY),
                               ("White", Colour.WHITE), ("", None), ("beige", None), ("black and white", None)):
            with self.subTest(answer=answer):
                self.assertIs(colour, colour_of_word(answer))


class ThePixelsTests(unittest.TestCase):
    def test_a_grey_part_agrees_with_grey(self) -> None:
        verdict = _verdict(GREY, Colour.GREY)
        self.assertTrue(verdict.agrees, verdict.why)
        self.assertFalse(verdict.unsure)

    def test_white_green_orange_and_black_are_refused_when_grey_is_wanted(self) -> None:
        for colour, seen in ((WHITE, "white"), (GREEN, "green"), (ORANGE, "orange"), (MAT, "black")):
            with self.subTest(seen=seen):
                verdict = _verdict(colour, Colour.GREY)
                self.assertFalse(verdict.agrees, verdict.why)
                self.assertFalse(verdict.unsure, verdict.why)
                self.assertEqual(seen, verdict.seen)

    def test_a_lightness_between_grey_and_white_is_unsure(self) -> None:
        """p80 L* 77 lies between the grey parts (52-69) and the white ones (80-100): the pixels cannot tell."""
        for wanted in (Colour.GREY, Colour.WHITE):
            with self.subTest(wanted=wanted):
                verdict = _verdict(_bgr_of_lab(77.0, 2.0, 90.0), wanted)
                self.assertTrue(verdict.unsure, verdict.why)
                self.assertIn("p80 L* 77", verdict.why)

    def test_fewer_than_fifty_pixels_is_unsure(self) -> None:
        """A 10 px square holds 36 px after the 2 px erosion."""
        verdict = _verdict(GREY, Colour.GREY, size=10, rim=0)
        self.assertTrue(verdict.unsure, verdict.why)
        self.assertIn("fewer than 50", verdict.why)

    def test_the_mat_the_mask_bled_in_is_left_out(self) -> None:
        """A rim of the mat 6 px wide inside the mask: the erosion takes 2, the dark rule the rest."""
        verdict = _verdict(GREY, Colour.GREY, rim=6)
        self.assertTrue(verdict.agrees, verdict.why)

    def test_a_colour_with_a_hue_agrees_in_its_band_and_is_refused_by_another(self) -> None:
        self.assertTrue(_verdict(GREEN, Colour.GREEN).agrees)
        self.assertTrue(_verdict(RED, Colour.RED).agrees, "the red the pick prompt tests paint")
        self.assertTrue(_verdict(ORANGE, Colour.ORANGE).agrees)
        for colour, wanted, seen in ((ORANGE, Colour.GREEN, "orange"), (GREEN, Colour.RED, "green"),
                                     (GREY, Colour.GREEN, "grey"), (WHITE, Colour.RED, "white")):
            with self.subTest(wanted=wanted, seen=seen):
                verdict = _verdict(colour, wanted)
                self.assertFalse(verdict.agrees or verdict.unsure, verdict.why)
                self.assertEqual(seen, verdict.seen)

    def test_white_and_black_are_judged_as_grey_is(self) -> None:
        self.assertTrue(_verdict(WHITE, Colour.WHITE).agrees)
        self.assertEqual("grey", _verdict(GREY, Colour.WHITE).seen)
        self.assertTrue(_verdict(MAT, Colour.BLACK).agrees, "black wanted: the dark pixels are counted")
        self.assertFalse(_verdict(GREY, Colour.BLACK).agrees)

    def test_a_part_dark_in_a_hue_s_place_is_asked_not_refused(self) -> None:
        """A dark part may be a dark shade of the colour wanted: the VLM is asked."""
        self.assertTrue(_verdict(MAT, Colour.GREEN).unsure)

    def test_pink_and_brown_are_always_asked(self) -> None:
        for wanted in (Colour.PINK, Colour.BROWN):
            with self.subTest(wanted=wanted):
                self.assertTrue(_verdict(_bgr_of_lab(60.0, 40.0, 10.0), wanted).unsure)


class TheVerdictTests(unittest.TestCase):
    def test_an_unsure_part_the_vlm_names_the_wanted_colour_of_agrees(self) -> None:
        unsure = _verdict(_bgr_of_lab(77.0, 2.0, 90.0), Colour.GREY)
        for answer in ("grey", "Gray.", "silver"):
            with self.subTest(answer=answer):
                verdict = unsure.answered(answer)
                self.assertTrue(verdict.agrees)
                self.assertFalse(verdict.unsure)
                self.assertEqual(answer, verdict.asked)

    def test_another_colour_word_no_colour_word_and_no_answer_refuse_it(self) -> None:
        unsure = _verdict(_bgr_of_lab(77.0, 2.0, 90.0), Colour.GREY)
        for answer, seen in (("white", "white"), ("beige", UNNAMED), ("", UNNAMED)):
            with self.subTest(answer=answer):
                verdict = unsure.answered(answer)
                self.assertFalse(verdict.agrees or verdict.unsure)
                self.assertEqual(seen, verdict.seen)

    def test_a_verdict_the_pixels_gave_is_kept_whatever_the_vlm_would_say(self) -> None:
        refused = _verdict(GREEN, Colour.GREY)
        self.assertIs(refused, refused.answered("grey"))
        agreed = _verdict(GREY, Colour.GREY)
        self.assertIs(agreed, agreed.answered("white"))

    def test_a_refused_part_goes_by_a_label_no_object_carries(self) -> None:
        self.assertEqual("green object, not grey cube", refused_label(_verdict(GREEN, Colour.GREY), "grey cube"))
        nothing = _verdict(_bgr_of_lab(77.0, 2.0, 90.0), Colour.GREY).answered("")
        self.assertEqual("unknown colour object, not grey cube", refused_label(nothing, "grey cube"))

    def test_the_log_line_says_what_was_wanted_seen_and_answered(self) -> None:
        said = _verdict(GREEN, Colour.GREY).said("grey cube")
        self.assertIn("grey wanted", said)
        self.assertIn("refuses it as green", said)
        asked = _verdict(_bgr_of_lab(77.0, 2.0, 90.0), Colour.GREY).answered("").said("grey cube")
        self.assertIn("no colour word came back", asked)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
