"""The stereo depth of a real pair agrees with the D415's own depth of it, within what was measured.

``tests/data/stereo/d415_ir_pair.npz`` is one look of the owner's Thursday picks (2026-10-15), cropped to the parts in
the bin: both infrared images the D415 handed out, rectified by the device, the depth its chip matched on them before
any filter, and the two lenses and their extrinsics (the file's ``source`` says which look and which crop). Its pair
goes through ``StereoDepth`` as a webcam pair's does in the pick path, its calibration built from the lenses by
``cv.stereoRectify``, and its depth is held to the chip's within the bounds measured on it with margin.
``scripts/stereo/measure_on_the_d415_ir_pair.py`` measures the same over every look of that day.

A crop is narrower than the frame, and its left 70 or so columns match into the right image left of the crop, where
nothing is: those are holes, so this crop holds depth on about 63 % of its pixels where the chip holds it on 94 %.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from src.calibration.stereo.config import CalibrationResult
from src.calibration.stereo.sub_modules.depth import StereoDepth, working_depth_range_mm
from src.calibration.stereo.sub_modules.disparity import DisparitySearch

_PAIR = Path(__file__).resolve().parent / "data" / "stereo" / "d415_ir_pair.npz"


def _measured() -> tuple[np.ndarray, np.ndarray, StereoDepth]:
    with np.load(_PAIR, allow_pickle=False) as data:
        left, right = data["ir_left"], data["ir_right"]
        chip = data["chip_depth_mm"].astype(np.float64)
        lenses, distortion, extrinsics = data["ir_intrinsics"], data["ir_distortion"], data["ir_extrinsics"]
    size = (int(left.shape[1]), int(left.shape[0]))
    rect_l, rect_r, proj_l, proj_r, q, _, _ = cv.stereoRectify(
        lenses[0], distortion[0], lenses[1], distortion[1], size, extrinsics[1][:3, :3], extrinsics[1][:3, 3],
        alpha=0.0)
    calibration = CalibrationResult(camL=lenses[0], distL=distortion[0], rectL=rect_l, projL=proj_l, camR=lenses[1],
                                    distR=distortion[1], rectR=rect_r, projR=proj_r, Q=q, frame_size=size)
    depth = StereoDepth(calibration, working_depth_range_mm())
    rectified = depth.rectify(left, right)
    return depth.depth_mm(*rectified).astype(np.float64), chip, depth


class TheD415PairTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.depth, cls.chip, cls.stereo = _measured()

    def test_the_file_is_small_and_says_where_it_comes_from(self) -> None:
        self.assertLess(_PAIR.stat().st_size, 300_000)
        with np.load(_PAIR, allow_pickle=False) as data:
            self.assertIn("look", str(data["source"]))

    def test_the_search_is_derived_from_the_d415s_own_lens_and_baseline(self) -> None:
        self.assertAlmostEqual(54.79, self.stereo.baseline_mm, places=2)
        self.assertAlmostEqual(894.96, self.stereo.focal_px, places=2)
        self.assertEqual(DisparitySearch(0, 336), self.stereo.search)

    def test_it_reads_most_of_what_the_chip_reads(self) -> None:
        measured, chip = self.depth > 0, self.chip > 0
        self.assertGreater(measured.mean(), 0.58)  # measured 0.633
        self.assertGreater((measured & chip).sum() / chip.sum(), 0.61)  # measured 0.665

    def test_where_both_read_they_agree_to_a_few_millimetres(self) -> None:
        both = (self.depth > 0) & (self.chip > 0)
        error = self.depth[both] - self.chip[both]
        self.assertLessEqual(float(np.median(np.abs(error))), 3.0)  # measured 2.0 mm, at 640 to 800 mm
        self.assertLessEqual(float(np.percentile(np.abs(error), 90)), 8.0)  # measured 6.0 mm
        self.assertLess(abs(float(np.mean(error))), 2.0)  # measured -0.9 mm: no scale off the chip's

    def test_it_invents_next_to_no_depth(self) -> None:
        both = (self.depth > 0) & (self.chip > 0)
        error = self.depth - self.chip
        limit = np.maximum(10.0, 0.03 * self.chip)
        self.assertLess(float((both & (error > limit)).sum() / both.sum()), 0.01)  # measured 0.44 %: read too far
        self.assertLess(float((both & (error < -limit)).sum() / both.sum()), 0.01)  # measured 0.03 %: too near


if __name__ == "__main__":
    unittest.main()
