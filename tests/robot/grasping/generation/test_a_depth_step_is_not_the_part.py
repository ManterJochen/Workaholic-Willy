"""A mask pixel that measures far behind its own neighbours sees past the part, not the part (2026-09-23).

``pixels_behind_depth_steps`` is the image-side half of the tilted-view fix: the Located path and the calculator's
support-footprint input both take the mask less these pixels. The scene-level measurements are in
``test_a_tilted_view_grasps_on_the_part.py``; these are the rule's own edges.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.generation.depth_steps import (
    DEPTH_STEP_MM,
    DEPTH_STEP_RADIUS_PX,
    pixels_behind_depth_steps,
)


def _part(rows: int = 20, cols: int = 20, depth_mm: float = 500.0) -> tuple[np.ndarray, np.ndarray]:
    """A 60x60 image with a flat part under a mask in the middle, 100 mm in front of the table."""
    depth = np.full((60, 60), depth_mm + 100.0)
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:20 + rows, 20:20 + cols] = True
    depth[mask] = depth_mm
    return mask, depth


class TheBackgroundInsideTheMaskIsFoundTests(unittest.TestCase):
    def test_rows_of_table_depth_at_the_masks_edge_are_found_and_nothing_else(self) -> None:
        mask, depth = _part()
        for rows in (1, 2, DEPTH_STEP_RADIUS_PX):
            leaked = depth.copy()
            leaked[20:20 + rows, 20:40] = 600.0          # misregistered: the table's depth under the top rows
            with self.subTest(rows=rows):
                behind = pixels_behind_depth_steps(mask, leaked)
                expected = np.zeros_like(mask)
                expected[20:20 + rows, 20:40] = True
                np.testing.assert_array_equal(expected, behind)

    def test_a_mask_with_no_step_inside_it_loses_nothing(self) -> None:
        mask, depth = _part()
        ramp = depth + np.linspace(0.0, 30.0, 60)[:, None]    # a surface seen obliquely: 0.5 mm per pixel
        self.assertFalse(pixels_behind_depth_steps(mask, ramp).any())

    def test_a_step_no_larger_than_the_threshold_is_a_surface(self) -> None:
        mask, depth = _part()
        stepped = depth.copy()
        stepped[20:22, 20:40] += DEPTH_STEP_MM
        self.assertFalse(pixels_behind_depth_steps(mask, stepped).any())
        stepped[20:22, 20:40] += 0.5
        self.assertTrue(pixels_behind_depth_steps(mask, stepped)[20:22, 20:40].all())

    def test_the_nearest_pixels_always_stay(self) -> None:
        """Half near, half far: the far half loses the band next to the step and keeps the rest."""
        mask, depth = _part()
        split = depth.copy()
        split[20:40, 30:40] = 560.0
        behind = pixels_behind_depth_steps(mask, split)
        self.assertFalse(behind[:, :30].any(), "the near side is the part")
        self.assertTrue(behind[20:40, 30:30 + DEPTH_STEP_RADIUS_PX].all())
        self.assertFalse(behind[20:40, 30 + DEPTH_STEP_RADIUS_PX:].any())


class OnlyTheMaskIsComparedTests(unittest.TestCase):
    def test_an_occluder_outside_the_mask_removes_nothing(self) -> None:
        mask, depth = _part()
        occluded = depth.copy()
        occluded[10:20, 10:50] = 300.0                    # a nearer object right above the part, not in its mask
        self.assertFalse(pixels_behind_depth_steps(mask, occluded).any())

    def test_pixels_without_depth_are_left_alone(self) -> None:
        mask, depth = _part()
        holed = depth.copy()
        holed[25:28, 25:28] = 0.0
        holed[30, 30] = np.nan
        self.assertFalse(pixels_behind_depth_steps(mask, holed).any())

    def test_an_empty_mask_and_a_mask_with_no_depth_give_nothing(self) -> None:
        mask, depth = _part()
        self.assertFalse(pixels_behind_depth_steps(np.zeros_like(mask), depth).any())
        self.assertFalse(pixels_behind_depth_steps(mask, np.zeros_like(depth)).any())

    def test_a_uint8_mask_and_uint16_depth_are_read_as_given(self) -> None:
        mask, depth = _part()
        leaked = depth.copy()
        leaked[20:22, 20:40] = 600.0
        np.testing.assert_array_equal(pixels_behind_depth_steps(mask, leaked),
                                      pixels_behind_depth_steps(mask.astype(np.uint8), leaked.astype(np.uint16)))


class ItRefusesWhatItCannotReadTests(unittest.TestCase):
    def test_shapes_and_knobs_are_checked(self) -> None:
        mask, depth = _part()
        with self.assertRaises(ValueError):
            pixels_behind_depth_steps(mask, depth[:, :-1])
        with self.assertRaises(ValueError):
            pixels_behind_depth_steps(mask, depth, step_mm=0.0)
        with self.assertRaises(ValueError):
            pixels_behind_depth_steps(mask, depth, radius_px=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
