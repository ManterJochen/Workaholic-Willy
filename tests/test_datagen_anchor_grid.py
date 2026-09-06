"""Anchors over the free PLANE, not along one line in it.

`_anchors` walks the longest free axis and offers a handful of points along it. On a large object
that line stays where the object is WIDEST, which is exactly where the jaw does not fit, so the
object earns no label at all however many approaches are tried against it.

MEASURED on 532 closable meshes, each screened alone and upright by the real `check_jaw_grasp`, not
by a proxy: jaw-graspable objects 148 -> 341 (x2.30), and the ones the grid gains are the LARGE
objects the centroid line could never fit, median longest extent 191 mm against 119 mm. Supervised
contact points x6.2 (30,118 -> 186,312), distinct 20 mm contact cells x3.7.

It is a THIRD density preset rather than a change to `dense`, because every corpus on this box was
labelled with `dense` and stamped with `dense`'s fields. Moving the grid into it would make two
different label sets answer to the same name, and the stamp is how a reader tells them apart.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.grasps.labels import DEFAULT_DENSITY, DENSITIES, GRID_DENSITY, _anchors
from datagen.grasps.shapes import Solid


def _box(half=(60.0, 40.0, 20.0)) -> Solid:
    return Solid(kind="box", half_extent_mm=np.asarray(half, dtype=float),
                 rotation=np.eye(3), centre_mm=np.zeros(3))


class TheDefaultIsUntouchedTests(unittest.TestCase):
    def test_the_two_shipped_presets_do_not_grid(self) -> None:
        for name in ("default", "dense"):
            with self.subTest(density=name):
                self.assertFalse(DENSITIES[name].anchor_grid)

    def test_the_shipped_anchors_are_byte_identical(self) -> None:
        """The exact points `dense` produced before the grid existed, for a box closing along z."""
        got = np.asarray(_anchors(_box(), np.array([0.0, 0.0, 1.0]), DENSITIES["dense"]))
        want = np.array([[0.0, 0.0, 0.0], [-12.0, 0.0, 0.0], [12.0, 0.0, 0.0],
                         [-24.0, 0.0, 0.0], [24.0, 0.0, 0.0]])
        np.testing.assert_allclose(got, want)

    def test_a_sphere_still_has_exactly_one_anchor(self) -> None:
        sphere = Solid(kind="sphere", half_extent_mm=np.array([30.0, 30.0, 30.0]),
                       rotation=np.eye(3), centre_mm=np.zeros(3))
        self.assertEqual(len(_anchors(sphere, np.array([0.0, 0.0, 1.0]), GRID_DENSITY)), 1)


class TheGridCoversThePlaneTests(unittest.TestCase):
    AXIS = np.array([0.0, 0.0, 1.0])          # closing along z leaves x and y free

    def test_it_is_the_product_of_the_fractions_on_both_free_axes(self) -> None:
        n = len(GRID_DENSITY.anchor_fractions)
        self.assertEqual(len(_anchors(_box(), self.AXIS, GRID_DENSITY)), n * n)

    def test_it_moves_along_the_axis_the_line_never_touched(self) -> None:
        """The whole point. The line varies one coordinate; the grid has to vary two."""
        line = np.asarray(_anchors(_box(), self.AXIS, DENSITIES["dense"]))
        grid = np.asarray(_anchors(_box(), self.AXIS, GRID_DENSITY))
        self.assertEqual(len(np.unique(np.round(line[:, 1], 6))), 1,
                         "the line already spread across y, so this test proves nothing")
        self.assertGreater(len(np.unique(np.round(grid[:, 1], 6))), 1)

    def test_it_never_leaves_the_closing_axis(self) -> None:
        """An anchor displaced ALONG the closing axis is a different grasp, not a wider search."""
        grid = np.asarray(_anchors(_box(), self.AXIS, GRID_DENSITY))
        np.testing.assert_allclose(grid @ self.AXIS, 0.0, atol=1e-9)

    def test_the_centre_is_still_among_the_anchors(self) -> None:
        """The line's best point must survive, or the grid trades reach for the easy grasps."""
        grid = _anchors(_box(), self.AXIS, GRID_DENSITY)
        self.assertTrue(any(np.allclose(p, np.zeros(3)) for p in grid))

    def test_the_line_is_a_subset_of_the_grid(self) -> None:
        line = np.asarray(_anchors(_box(), self.AXIS, DENSITIES["dense"]))
        grid = np.asarray(_anchors(_box(), self.AXIS, GRID_DENSITY))
        for point in line:
            self.assertTrue(any(np.allclose(point, g) for g in grid),
                            f"the grid dropped a point the line had: {point}")

    def test_one_free_axis_falls_back_to_the_line(self) -> None:
        """A flat slab whose closing axis pins two axes has no plane to grid over."""
        slab = Solid(kind="box", half_extent_mm=np.array([60.0, 40.0, 20.0]),
                     rotation=np.eye(3), centre_mm=np.zeros(3))
        # An axis 45 degrees between x and y leaves only z clearly free.
        axis = np.array([0.7071067811865476, 0.7071067811865476, 0.0])
        anchors = _anchors(slab, axis, GRID_DENSITY)
        self.assertLessEqual(len(anchors), len(GRID_DENSITY.anchor_fractions) ** 2)


class TheStampSaysWhichAnchoringWasUsedTests(unittest.TestCase):
    def test_the_report_row_carries_the_flag(self) -> None:
        """A label COUNT means nothing without it: the two presets differ by a factor of five."""
        self.assertIn("anchor_grid", DEFAULT_DENSITY.as_row())
        self.assertFalse(DEFAULT_DENSITY.as_row()["anchor_grid"])
        self.assertTrue(GRID_DENSITY.as_row()["anchor_grid"])

    def test_the_cli_offers_it(self) -> None:
        import inspect

        from datagen import __main__ as cli
        src = inspect.getsource(cli)
        self.assertIn('choices=("default", "dense", "grid")', src,
                      "the density exists but no command line can ask for it")
