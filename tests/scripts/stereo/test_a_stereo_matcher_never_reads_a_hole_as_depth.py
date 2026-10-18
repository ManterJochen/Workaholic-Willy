"""The stereo matcher reads no depth it did not match: a hole is a hole, at every search and after every frame.

The guard judges every motion against a stereo rig's depth, so the matcher is held to what an RGB-D camera's depth is
held to (the owner, 2026-10-17): a pixel it cannot match carries no depth, nothing is filled in from the neighbours or
from an earlier frame, and the settings are the measured ones, not a block of YAML.

⛔ MEASURED BEFORE THIS FILE: the matcher marked a pixel invalid where its disparity was not above 0, and OpenCV marks
an unmatched pixel ``(minDisparity - 1) * 16``. With a search that starts above 1 every unmatched pixel came out as a
disparity one short of the search's start, a depth near its far end that no surface stands at.
"""

from __future__ import annotations

import unittest

import cv2 as cv
import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.stereo.sub_modules.disparity import (
    MEASURED_SETTINGS,
    DisparityComputer,
    DisparitySearch,
    MatcherSettings,
)


def _texture(shape: tuple[int, int], seed: int = 3) -> np.ndarray:
    """A grey speckle texture with structure at a few pixels, as a projector's dots or a print give a matcher."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.0, 255.0, size=(shape[0] // 2 + 1, shape[1] // 2 + 1)).astype(np.float32)
    grown = cv.resize(noise, (shape[1], shape[0]), interpolation=cv.INTER_LINEAR)
    return np.clip(cv.GaussianBlur(grown, (0, 0), 0.8), 0, 255).astype(np.uint8)


def _shifted_pair(disparity_px: int, shape: tuple[int, int] = (120, 320)) -> tuple[np.ndarray, np.ndarray]:
    """A rectified pair of one plane at ``disparity_px``: a left pixel ``x`` is the right one at ``x - disparity_px``."""
    height, width = shape
    wide = _texture((height, width + disparity_px))
    return np.ascontiguousarray(wide[:, :width]), np.ascontiguousarray(wide[:, disparity_px:disparity_px + width])


class AHoleIsAHoleTests(unittest.TestCase):

    def test_a_search_that_starts_above_one_marks_no_unmatched_pixel_as_matched(self) -> None:
        blank = np.full((96, 256), 128, dtype=np.uint8)  # nothing to match anywhere
        computer = DisparityComputer(DisparitySearch(32, 64))
        raw = computer._matcher.compute(blank, blank)  # noqa: SLF001 (OpenCV's own answer, beside ours)
        self.assertTrue(np.all(raw == (32 - 1) * 16), "OpenCV marks an unmatched pixel (minDisparity - 1) * 16")
        disparity, valid = computer.compute(blank, blank)
        self.assertFalse(valid.any())
        self.assertTrue(np.all(np.isnan(disparity)))

    def test_a_textured_plane_is_read_at_its_disparity(self) -> None:
        left, right = _shifted_pair(40)
        disparity, valid = DisparityComputer(DisparitySearch(0, 64)).compute(left, right)
        self.assertGreater(valid.mean(), 0.5)
        self.assertLess(float(np.nanmax(np.abs(disparity[valid] - 40.0))), 0.5)
        self.assertTrue(np.all(np.isnan(disparity[~valid])))

    def test_a_surface_without_texture_holds_no_depth_carried_in_from_around_it(self) -> None:
        """A box of one grey level on a textured plane: what the matcher reads on its face is the plane's depth."""
        height, width = 160, 400
        left, right = _shifted_pair(20, (height, width))
        left[40:120, 150:250] = 128  # the box's face, at disparity 44: in the right image 44 pixels further left
        right[40:120, 150 - 44:250 - 44] = 128
        face = np.zeros((height, width), dtype=bool)
        face[44:116, 154:246] = True  # every block of 5 about these pixels lies on the face
        search = DisparitySearch(0, 64)
        kept, kept_valid = DisparityComputer(search, MatcherSettings(refuse_flat_blocks=False)).compute(left, right)
        self.assertTrue(np.any(kept_valid & face), "kept, a flat block holds a depth")
        self.assertLess(float(np.max(np.abs(kept[kept_valid & face] - 20.0))), 1.0, "the plane's, behind the face")
        self.assertFalse(np.any(DisparityComputer(search).compute(left, right)[1] & face))

    def test_a_plane_beyond_the_far_end_of_a_search_is_no_depth_inside_it(self) -> None:
        left, right = _shifted_pair(12)  # a plane at disparity 12, behind a search that starts at 32
        _, valid = DisparityComputer(DisparitySearch(32, 64)).compute(left, right)
        self.assertLess(valid.mean(), 0.05)

    def test_what_is_found_does_not_depend_on_the_frames_before(self) -> None:
        computer = DisparityComputer(DisparitySearch(0, 64))
        first_left, first_right = _shifted_pair(20)
        left, right = _shifted_pair(44)
        alone = computer.compute(left, right)
        computer.compute(first_left, first_right)
        after = computer.compute(left, right)
        np.testing.assert_array_equal(alone[1], after[1])
        np.testing.assert_array_equal(np.nan_to_num(alone[0], nan=-1.0), np.nan_to_num(after[0], nan=-1.0))

    def test_a_pixel_beside_a_hole_or_a_step_is_a_hole_too(self) -> None:
        height, width = 120, 320
        far_left, far_right = _shifted_pair(20, (height, width))
        near_left, _ = _shifted_pair(44, (height, width))
        left, right = far_left.copy(), far_right.copy()
        # A near rectangle over the far plane: a disparity step of 24 along its edges, an occluded strip beside it.
        left[40:80, 150:230] = near_left[40:80, 150:230]
        right[40:80, 150 - 44:230 - 44] = near_left[40:80, 150:230]
        search = DisparitySearch(0, 64)
        disparity, valid = DisparityComputer(search).compute(left, right)
        raw_disparity, raw_valid = DisparityComputer(
            search, MatcherSettings(hole_margin_px=0, step_margin_px=0)).compute(left, right)
        self.assertFalse(np.any(valid & ~raw_valid), "the margins only take depth away")
        step = np.zeros((height, width), dtype=bool)
        across = raw_valid[:, 1:] & raw_valid[:, :-1] & (
            np.abs(raw_disparity[:, 1:] - raw_disparity[:, :-1]) > MEASURED_SETTINGS.step_px)
        step[:, 1:] |= across
        step[:, :-1] |= across
        self.assertTrue(step.any() and (~raw_valid).any(), "the scene has a step and a hole to keep away from")
        self.assertFalse(np.any(valid & _within(~raw_valid, MEASURED_SETTINGS.hole_margin_px)))
        self.assertFalse(np.any(valid & _within(step, MEASURED_SETTINGS.step_margin_px)))
        self.assertGreater(float(np.nanmedian(disparity[45:75, 160:220])), 43.5)  # the rectangle itself is read
        self.assertLess(float(np.nanmedian(disparity[90:115, 20:300])), 20.5)  # and the plane behind it

    def test_the_left_band_is_read_where_its_match_lies_inside_the_right_image(self) -> None:
        left, right = _shifted_pair(40, (96, 320))
        padded = DisparityComputer(DisparitySearch(0, 128)).compute(left, right)[1]
        unpadded = DisparityComputer(DisparitySearch(0, 128), MatcherSettings(pad_left=False)).compute(left, right)[1]
        self.assertFalse(unpadded[:, :128].any(), "unpadded, the band the search is wide holds no depth at all")
        self.assertGreater(padded[:, 48:128].mean(), 0.8, "padded, a pixel whose match is in the right image is read")
        self.assertFalse(padded[:, :38].any(), "a pixel whose match would lie left of the right image is not")


def _within(seeds: np.ndarray, margin: int) -> np.ndarray:
    """The pixels within ``margin`` of a seed, the square about it, as the margins read them."""
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    return cv.dilate(seeds.astype(np.uint8), kernel, borderType=cv.BORDER_CONSTANT, borderValue=0) > 0


class TheSettingsAreTheMeasuredOnesTests(unittest.TestCase):
    """The values ``scripts/stereo/measure_on_the_d415_ir_pair.py`` chose. A change here is a change to measure again."""

    def test_the_measured_settings(self) -> None:
        self.assertEqual(
            MatcherSettings(block_size=5, smoothness_p1=8, smoothness_p2=32, uniqueness_ratio=15, speckle_window_size=100,
                            speckle_range=2, disp12_max_diff=1, three_way=True, pad_left=True, left_right_px=1.0,
                            near_end_px=1, refuse_flat_blocks=True, hole_margin_px=2, step_margin_px=2, step_px=1.0),
            MEASURED_SETTINGS)

    def test_the_matcher_is_built_with_them(self) -> None:
        matcher = DisparityComputer(DisparitySearch(16, 96))._matcher  # noqa: SLF001
        block = MEASURED_SETTINGS.block_size
        self.assertEqual((16, 96, block), (matcher.getMinDisparity(), matcher.getNumDisparities(), matcher.getBlockSize()))
        self.assertEqual((8 * block * block, 32 * block * block), (matcher.getP1(), matcher.getP2()))
        self.assertEqual(cv.STEREO_SGBM_MODE_SGBM_3WAY, matcher.getMode())
        self.assertEqual((15, 100, 2, 1), (matcher.getUniquenessRatio(), matcher.getSpeckleWindowSize(),
                                           matcher.getSpeckleRange(), matcher.getDisp12MaxDiff()))

    def test_settings_that_build_no_matcher_are_refused(self) -> None:
        for bad in (dict(block_size=4), dict(block_size=0), dict(smoothness_p1=40, smoothness_p2=32),
                    dict(uniqueness_ratio=-1), dict(hole_margin_px=-1), dict(left_right_px=-0.5)):
            with self.subTest(bad), self.assertRaises(StereoCalibrationError):
                MatcherSettings(**bad)  # type: ignore[arg-type]

    def test_a_pair_of_two_sizes_or_of_another_kind_is_refused(self) -> None:
        computer = DisparityComputer(DisparitySearch(0, 16))
        with self.assertRaises(StereoCalibrationError):
            computer.compute(np.zeros((10, 20), np.uint8), np.zeros((10, 21), np.uint8))
        with self.assertRaises(StereoCalibrationError):
            computer.compute(np.zeros((10, 20, 4), np.uint8), np.zeros((10, 20, 4), np.uint8))


if __name__ == "__main__":
    unittest.main()
