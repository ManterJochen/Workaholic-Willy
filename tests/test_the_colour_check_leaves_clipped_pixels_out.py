"""The colour check leaves out the pixels the camera clipped in some channels and not all (the owner, 2026-10-09).

The colour camera runs on auto exposure. Where the mat around an orange part is dark it clips the part's red channel,
so the part's hue is read off its green channel alone: BGR (40, 191, 252), the orange the review measured at the
brighter look poses, reads hue 82.5 degrees, in the yellow band. ``robot.grasping.colour_check_clipped``: ``exclude``,
the owner's choice as shipped ("wenn das schon so gut war, dann nutzen wir das doch instant"), leaves every pixel with
one or two channels at 250 or over out of the counts; ``keep`` counts them, as before. Measured offline on the 31 looks
the cell recorded on 2026-10-07: orange judged right 53 of 89 times with them counted, 89 of 89 with them left out,
and the grey cubes, the white and green parts and the yellow bin judged as before. No red part stood in those views.

The patches are 40 px parts on the owner's mat (L* 16), the mask the part: an orange part whose upper rows clip and
whose lowest rows, in the shade, do not (BGR 21, 111, 242, hue 55); the grey, white and green parts of
``tests/test_a_part_of_another_colour_is_no_target.py``; the yellow bin, a quarter of it clipped as the review found it.
"""

from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from src.robot.perception.colour_check import CLIPPED_AT, Colour, ColourClipped, ColourVerdict, judge_colour

MAT = (34, 40, 42)
GREY = (156, 160, 162)
WHITE = (236, 240, 242)
GREEN = (117, 157, 101)
#: An orange part in the shade, every channel under the clip: hue 55.
ORANGE = (21, 111, 242)
#: The same part where the camera clipped its red channel: hue 82.5, the yellow band.
CLIPPED_ORANGE = (40, 191, 252)
#: The yellow bin, and where its red channel clipped.
YELLOW, CLIPPED_YELLOW = (39, 182, 214), (40, 230, 255)
#: Glare on a white part: every channel clipped.
GLARE = (255, 255, 255)


def _part(rows: list[tuple[int, tuple[int, int, int]]], *, mat_rim: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A 40 px part at (20, 20) on the mat, painted top to bottom by ``(how many rows, BGR)``; its mask the whole part,
    the outer ``mat_rim`` px of it the mat's, as a mask's edge bleeds the mat in."""
    image = np.empty((80, 80, 3), dtype=np.uint8)
    image[...] = MAT
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:60, 20:60] = True
    row = 20
    for count, colour in rows:
        image[row:row + count, 20:60] = colour
        row += count
    assert row == 60, "the rows paint the part's 40 px"
    if mat_rim:
        inner = np.zeros_like(mask)
        inner[20 + mat_rim:60 - mat_rim, 20 + mat_rim:60 - mat_rim] = True
        image[mask & ~inner] = MAT
    return image, mask


def _both(image: np.ndarray, mask: np.ndarray, wanted: Colour) -> tuple[ColourVerdict, ColourVerdict]:
    return (judge_colour(image, mask, wanted, clipped=ColourClipped.KEEP),
            judge_colour(image, mask, wanted, clipped=ColourClipped.EXCLUDE))


class AClippedOrangePartTests(unittest.TestCase):
    """34 rows clipped and the 6 lowest in the shade: after the 2 px erosion 144 of its 1296 px are under the clip."""

    def _part(self) -> tuple[np.ndarray, np.ndarray]:
        return _part([(34, CLIPPED_ORANGE), (6, ORANGE)])

    def test_with_the_clipped_pixels_left_out_it_is_orange(self) -> None:
        verdict = judge_colour(*self._part(), Colour.ORANGE, clipped="exclude")
        self.assertTrue(verdict.agrees, verdict.why)
        self.assertEqual("orange", verdict.seen)
        self.assertIn("144 px, 1152 clipped left out", verdict.why)

    def test_with_them_kept_it_is_refused_as_yellow_as_before(self) -> None:
        verdict = judge_colour(*self._part(), Colour.ORANGE, clipped="keep")
        self.assertFalse(verdict.agrees or verdict.unsure, verdict.why)
        self.assertEqual("yellow", verdict.seen)
        self.assertNotIn("clipped", verdict.why)

    def test_the_owners_choice_is_the_default(self) -> None:
        image, mask = self._part()
        self.assertEqual(judge_colour(image, mask, Colour.ORANGE, clipped=ColourClipped.EXCLUDE),
                         judge_colour(image, mask, Colour.ORANGE))

    def test_a_clip_is_some_channels_at_250_and_not_all(self) -> None:
        self.assertEqual(250, CLIPPED_AT)
        for red, clipped in ((249, False), (250, True)):
            with self.subTest(red=red):
                image, mask = _part([(34, (40, 191, red)), (6, ORANGE)])
                verdict = judge_colour(image, mask, Colour.ORANGE)
                self.assertEqual(clipped, "clipped left out" in verdict.why, verdict.why)


class TooFewPixelsLeftTests(unittest.TestCase):
    def test_a_part_left_with_fewer_than_fifty_is_unsure_and_the_vlm_is_asked(self) -> None:
        """The lowest 3 rows in the shade leave 36 px after the erosion."""
        image, mask = _part([(37, CLIPPED_ORANGE), (3, ORANGE)])
        keep, exclude = _both(image, mask, Colour.ORANGE)
        self.assertTrue(exclude.unsure, exclude.why)
        self.assertIn("fewer than 50", exclude.why)
        self.assertIn("1260 clipped left out", exclude.why)
        self.assertEqual("yellow", keep.seen, "as before, the clipped pixels counted")

    def test_a_part_whose_every_pixel_clipped_is_asked_not_judged(self) -> None:
        """The orange the presentation tests painted, BGR (32, 120, 252): its red channel clipped on every pixel."""
        image, mask = _part([(40, (32, 120, 252))])
        keep, exclude = _both(image, mask, Colour.ORANGE)
        self.assertTrue(exclude.unsure, exclude.why)
        self.assertTrue(keep.agrees, "as before, its hue read with the clip")

    def test_the_pixels_left_out_never_make_a_part_black(self) -> None:
        """A grey part wanted, and the part a clipped orange with a dark seam: of the pixels read, most are dark, of the
        part, most are clipped. It is unsure, not refused as black."""
        image, mask = _part([(28, CLIPPED_ORANGE), (12, MAT)])
        verdict = judge_colour(image, mask, Colour.GREY)
        self.assertTrue(verdict.unsure, verdict.why)
        self.assertNotEqual("black", verdict.seen)


class WhatNoClipTouchesIsJudgedAsBeforeTests(unittest.TestCase):
    def test_grey_white_green_and_the_mat_get_the_same_verdict_either_way(self) -> None:
        for colour in (GREY, WHITE, GREEN, MAT, ORANGE):
            for wanted in (Colour.GREY, Colour.WHITE, Colour.BLACK, Colour.GREEN, Colour.ORANGE):
                with self.subTest(colour=colour, wanted=wanted):
                    keep, exclude = _both(*_part([(40, colour)], mat_rim=4), wanted)
                    self.assertEqual(keep, exclude)

    def test_glare_clipped_in_every_channel_is_counted_either_way(self) -> None:
        image, mask = _part([(4, GLARE), (36, WHITE)])
        keep, exclude = _both(image, mask, Colour.WHITE)
        self.assertEqual(keep, exclude)
        self.assertTrue(exclude.agrees, exclude.why)

    def test_the_yellow_bin_stays_yellow(self) -> None:
        image, mask = _part([(10, CLIPPED_YELLOW), (30, YELLOW)])
        keep, exclude = _both(image, mask, Colour.YELLOW)
        self.assertTrue(keep.agrees, keep.why)
        self.assertTrue(exclude.agrees, exclude.why)

    def test_a_switch_that_names_no_rule_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            judge_colour(*_part([(40, GREY)]), Colour.GREY, clipped="sometimes")


class TheKeyTests(unittest.TestCase):
    def test_exclude_is_the_default_of_the_schema_and_of_the_repositorys_tree(self) -> None:
        from src.config.loader import load_robot_section

        self.assertEqual("exclude", RobotGraspingConfig().colour_check_clipped)
        self.assertEqual("exclude", load_robot_section().grasping.colour_check_clipped)

    def test_keep_loads_and_another_word_is_refused(self) -> None:
        self.assertEqual("keep", RobotGraspingConfig(colour_check_clipped="keep").colour_check_clipped)
        for wrong in ("on", True, "drop"):
            with self.subTest(wrong=wrong), self.assertRaises(ValidationError):
                RobotGraspingConfig(colour_check_clipped=wrong)

    def test_the_reference_tree_writes_it_down_with_its_default(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "config" / "all_keys" / "robot" / "robot.yaml").read_text(
            encoding="utf-8")
        before, _, after = text.partition("    colour_check_clipped: exclude\n")
        self.assertTrue(after, "robot.grasping.colour_check_clipped is not in config/all_keys/robot/robot.yaml")
        self.assertIn("# [one of: keep | exclude]", before.splitlines()[-1])
        self.assertIn("# [default: exclude]", before.splitlines()[-2])


# --------------------------------------------------------------------------- the live camera source, as built today


@dataclass
class _Det:
    box: tuple[float, float, float, float]
    label: str
    score: float = 0.9


@dataclass(frozen=True)
class _Seg:
    mask: np.ndarray
    label: str


class _Streamer:
    def __init__(self, color: np.ndarray) -> None:
        from src.camera.setup.image_taking.frames import RGBDFrame

        self._frame = RGBDFrame(color=color, depth=np.full(color.shape[:2], 500, np.uint16))

    def grab(self):  # noqa: ANN201
        return self._frame

    def get_intrinsics(self) -> np.ndarray:
        return np.array([[600.0, 0.0, 40.0], [0.0, 600.0, 40.0], [0.0, 0.0, 1.0]])


class _Detector:
    def __init__(self, dets: list[_Det]) -> None:
        self._dets = dets

    def detect_all(self, bgr: np.ndarray, prompt: str) -> list[_Det]:
        return self._dets


class _Segmenter:
    def segment_detection(self, bgr: np.ndarray, det: _Det) -> _Seg:
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        x0, y0, x1, y1 = (int(c) for c in det.box)
        mask[y0:y1, x0:x1] = 1
        return _Seg(mask=mask, label=det.label)


class TheCameraSourceTests(unittest.TestCase):
    def test_the_camera_source_reads_an_orange_part_whose_red_clipped_as_orange(self) -> None:
        """The source judges as shipped: the clipped orange part stays an orange cube, a target, and the log says how
        many pixels were left out."""
        from src.robot.perception import RealSenseVisionPerceptionSource

        image, _mask = _part([(34, CLIPPED_ORANGE), (6, ORANGE)])
        prompt = "each separate orange cube"
        source = RealSenseVisionPerceptionSource(
            streamer=_Streamer(image), detector=_Detector([_Det(box=(20, 20, 60, 60), label=prompt)]),
            segmenter=_Segmenter(), object_labels=("orange cube",), prompt=prompt, warmup_grabs=0)
        with self.assertLogs("src.robot.perception.realsense_source", level=logging.INFO) as said:
            frame = source.acquire()
        self.assertEqual(["orange cube"], [seg.label for seg in frame.segmentations])
        self.assertTrue(any("agrees" in line and "clipped left out" in line for line in said.output), said.output)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
