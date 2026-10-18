"""A stereo pair's matcher searches the disparities its own calibration gives for the depths it can see.

The owner, 2026-10-17: the disparity range is derived from the calibration (focal length, baseline) and the depth of the
working volume, and no YAML holds it. Two pure functions do it, and these hold each to what it promises: the search
reads every depth of its range, a sub-pixel neighbour past each end included, in OpenCV's whole sixteens
(``disparity_search``); and the range is where the camera's view of the work begins, a fixed camera's workspace box
as it sees it or the near limit, and runs to infinity, because the matcher gives a surface beyond its search a wrong
depth inside it (``working_depth_range_mm``, measured on the D415's pair).
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.stereo.sub_modules.depth import NEAR_LIMIT_MM, working_depth_range_mm
from src.calibration.stereo.sub_modules.disparity import MEASURED_SETTINGS, DisparitySearch, disparity_search

#: The D415's infrared pair, as the owner's views record it: focal length (px) and baseline (mm).
_D415 = (894.958, 54.794)


def _box(x=(-500.0, 500.0), y=(-500.0, 500.0), z=(0.0, 300.0)) -> SimpleNamespace:  # noqa: ANN001
    return SimpleNamespace(x_min=x[0], x_max=x[1], y_min=y[0], y_max=y[1], z_min=z[0], z_max=z[1])


def _looking_down_from(height_mm: float) -> np.ndarray:
    """CAMERA to BASE of a camera at (0, 0, height) whose optical axis points straight down the base z."""
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[2, 3] = height_mm
    return transform


class TheSearchReadsEveryDepthOfItsRangeTests(unittest.TestCase):

    def test_the_d415_pair_from_the_near_limit_to_infinity(self) -> None:
        self.assertEqual(DisparitySearch(0, 336), disparity_search(*_D415, NEAR_LIMIT_MM))

    def test_a_far_end_starts_the_search_one_pixel_short_of_it(self) -> None:
        focal, baseline = _D415
        search = disparity_search(focal, baseline, 250.0, 1364.0)
        self.assertEqual(math.floor(focal * baseline / 1364.0) - 1, search.min_disparity)
        self.assertEqual(DisparitySearch(34, 176), search)

    def test_every_depth_of_the_range_lies_inside_what_is_kept(self) -> None:
        for focal, baseline, near, far in ((894.958, 54.794, 150.0, math.inf), (600.0, 60.0, 300.0, 2000.0),
                                           (1400.0, 120.0, 900.0, 1500.0), (300.0, 10.0, 50.0, 60.0)):
            with self.subTest(focal=focal, baseline=baseline, near=near, far=far):
                search = disparity_search(focal, baseline, near, far)
                self.assertEqual(0, search.num_disparities % 16)
                kept_below = search.max_disparity - MEASURED_SETTINGS.near_end_px
                nearest = focal * baseline / near
                farthest = 0.0 if math.isinf(far) else focal * baseline / far
                # A sub-pixel fit needs the neighbour beyond each end inside the search too.
                self.assertLess(nearest + 1.0, kept_below)
                self.assertLessEqual(search.min_disparity, max(0.0, farthest - 1.0))

    def test_a_search_to_infinity_starts_at_zero(self) -> None:
        self.assertEqual(0, disparity_search(*_D415, 400.0, math.inf).min_disparity)
        self.assertEqual(0, disparity_search(*_D415, 400.0).min_disparity)

    def test_the_width_grows_as_the_near_end_comes_closer(self) -> None:
        widths = [disparity_search(*_D415, near).num_disparities for near in (900.0, 450.0, 300.0, 150.0)]
        self.assertEqual(sorted(widths), widths)

    def test_a_range_or_a_lens_that_reads_nothing_is_refused(self) -> None:
        for args in ((894.0, 55.0, 0.0, 100.0), (894.0, 55.0, 200.0, 200.0), (894.0, 55.0, 300.0, 200.0),
                     (0.0, 55.0, 150.0, math.inf), (894.0, -55.0, 150.0, math.inf), (894.0, math.nan, 150.0, 900.0),
                     (894.0, 55.0, math.inf, math.inf)):
            with self.subTest(args=args), self.assertRaises(StereoCalibrationError):
                disparity_search(*args)

    def test_a_search_says_what_opencv_takes(self) -> None:
        with self.assertRaises(StereoCalibrationError):
            DisparitySearch(0, 100)
        with self.assertRaises(StereoCalibrationError):
            DisparitySearch(-1, 64)
        self.assertEqual(96, DisparitySearch(32, 64).max_disparity)


class TheRangeIsWhereTheViewOfTheWorkBeginsTests(unittest.TestCase):

    def test_without_a_box_or_a_placement_it_is_the_near_limit_to_infinity(self) -> None:
        self.assertEqual((NEAR_LIMIT_MM, math.inf), working_depth_range_mm())
        self.assertEqual((NEAR_LIMIT_MM, math.inf), working_depth_range_mm(workspace=_box()))
        self.assertEqual((NEAR_LIMIT_MM, math.inf), working_depth_range_mm(camera_to_base=_looking_down_from(1200.0)))

    def test_a_fixed_camera_starts_at_the_nearest_corner_of_the_box_it_sees(self) -> None:
        near, far = working_depth_range_mm(workspace=_box(z=(0.0, 300.0)), camera_to_base=_looking_down_from(1200.0))
        self.assertAlmostEqual(900.0, near)  # 1200 above the base, the box's top 300 above it
        self.assertEqual(math.inf, far)  # the floor and the walls beyond the box are in view too

    def test_a_tilted_camera_reads_the_corners_along_its_own_axis(self) -> None:
        tilted = _looking_down_from(1000.0)
        angle = math.radians(30.0)
        turn = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(angle), -math.sin(angle)], [0.0, math.sin(angle), math.cos(angle)]])
        tilted[:3, :3] = tilted[:3, :3] @ turn
        box = _box(x=(-100.0, 100.0), y=(-100.0, 100.0), z=(0.0, 100.0))
        corners = np.array([[x, y, z] for x in (-100.0, 100.0) for y in (-100.0, 100.0) for z in (0.0, 100.0)])
        to_camera = np.linalg.inv(tilted)
        expected = min(float((to_camera[:3, :3] @ corner + to_camera[:3, 3])[2]) for corner in corners)
        near, _ = working_depth_range_mm(workspace=box, camera_to_base=tilted)
        self.assertAlmostEqual(expected, near, places=6)

    def test_a_camera_inside_its_box_starts_at_the_near_limit(self) -> None:
        near, far = working_depth_range_mm(workspace=_box(z=(0.0, 900.0)), camera_to_base=_looking_down_from(700.0))
        self.assertEqual((NEAR_LIMIT_MM, math.inf), (near, far))

    def test_a_box_nearer_than_the_near_limit_starts_at_the_limit(self) -> None:
        near, _ = working_depth_range_mm(workspace=_box(z=(0.0, 300.0)), camera_to_base=_looking_down_from(350.0))
        self.assertEqual(NEAR_LIMIT_MM, near)

    def test_a_placement_that_puts_the_whole_box_behind_the_camera_is_refused(self) -> None:
        looking_up = np.eye(4)
        looking_up[2, 3] = 1200.0  # its optical axis along +z, away from the box beneath it
        with self.assertRaisesRegex(StereoCalibrationError, "behind"):
            working_depth_range_mm(workspace=_box(z=(0.0, 300.0)), camera_to_base=looking_up)

    def test_a_box_without_volume_or_a_placement_that_is_no_transform_is_refused(self) -> None:
        with self.assertRaises(StereoCalibrationError):
            working_depth_range_mm(workspace=_box(z=(300.0, 300.0)), camera_to_base=_looking_down_from(1200.0))
        with self.assertRaises(StereoCalibrationError):
            working_depth_range_mm(workspace=_box(), camera_to_base=np.eye(3))


if __name__ == "__main__":
    unittest.main()
