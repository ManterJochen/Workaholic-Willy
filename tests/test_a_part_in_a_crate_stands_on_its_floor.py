"""A part held in a crate stands on the crate's floor, though the parts round it cut the floor into strips.

On the owner's cell (2026-10-12, 11:49 and 11:51) the camera world found no surface in the deep crate: the parts on its
floor cut it into strips narrower than the 75 mm a surface needs, so the world held the bench under the target, and the
crate's floor round it stood as obstacles. Where a held target stands on no surface found, the ground round it is read
now (``support_surfaces._ground_round``): the ring of pixels 5 to 50 mm round its box, under its top and over the
bench's band, whose lowest level holds 15 % of them on three of the eight sides round it or more, is a region the
surface builder holds to every rule a found surface keeps.

* the crate's floor under the held part is a support, at the floor's height; with no part held it is not, as before;
* a part beside one block, and a part on a block too small to be a surface, stand on no ground: a ledge on one side is
  no floor, and a part's top under another is never a support.

The scenes are drawn as ``tests/test_support_surfaces_are_found`` draws them: a D415 800 mm over the bench, looking
straight down.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.robot.core.keep_out import KeepOutBox
from tests.test_support_surfaces_are_found import _down, _found, _render, _view

#: The crate's inside floor, its walls' top, and the parts on its floor: 40 mm cubes on a 70 x 60 mm grid, 30 and 20 mm
#: of floor between them, as the owner's crate was cut on 2026-10-12.
FLOOR_MM = 70.0
WALL_MM = 211.0
CUBE_MM = 40.0
_CRATE = [((0.0, 0.0, FLOOR_MM / 2.0), (260.0, 200.0, FLOOR_MM), None),
          ((-134.0, 0.0, WALL_MM / 2.0), (8.0, 216.0, WALL_MM), None),
          ((134.0, 0.0, WALL_MM / 2.0), (8.0, 216.0, WALL_MM), None),
          ((0.0, -104.0, WALL_MM / 2.0), (260.0, 8.0, WALL_MM), None),
          ((0.0, 104.0, WALL_MM / 2.0), (260.0, 8.0, WALL_MM), None)]
_CUBES = [((x, y, FLOOR_MM + CUBE_MM / 2.0), (CUBE_MM, CUBE_MM, CUBE_MM), None)
          for x in (-70.0, 0.0, 70.0) for y in (-60.0, 0.0, 60.0)]


def _held(x: float, y: float, z_top: float, size: float = CUBE_MM) -> KeepOutBox:
    """The box a pick holds round its target: the part's footprint, from its top down its height, its own points
    reaching down to its foot."""
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z_top - size / 2.0)
    return KeepOutBox.from_matrix("target", matrix, (size / 2.0, size / 2.0, size / 2.0), low_mm=z_top - size)


def _model(boxes: Any, keep_out: Any = ()) -> Any:
    camera = _down()
    return _found([_view(camera, _render(camera, boxes=boxes, seed=11))], keep_out=keep_out)


def _height_under(model: Any, x: float, y: float) -> "float | None":
    value = model.height_under(np.array([[x, y]]))
    return None if value is None else float(value)


class AHeldPartInACrateStandsOnItsFloorTests(unittest.TestCase):
    def test_the_floor_under_the_held_part_is_a_support_at_its_height(self) -> None:
        model = _model(_CRATE + _CUBES, keep_out=[_held(0.0, 0.0, FLOOR_MM + CUBE_MM)])

        under = _height_under(model, 0.0, 0.0)
        assert under is not None, model.render()
        self.assertAlmostEqual(FLOOR_MM, under, delta=3.0)

    def test_with_no_part_held_the_cut_floor_is_no_support_as_before(self) -> None:
        model = _model(_CRATE + _CUBES)

        under = _height_under(model, 0.0, 0.0)
        self.assertTrue(under is None or under < FLOOR_MM - 20.0, model.render())


class NoGroundWhereThereIsNoneTests(unittest.TestCase):
    def test_a_part_beside_one_block_stands_on_no_ledge(self) -> None:
        """The block's top reaches round the part on one side: no floor under it."""
        block = ((70.0, 0.0, 30.0), (60.0, 200.0, 60.0), None)
        part = ((0.0, 0.0, CUBE_MM / 2.0 + 60.0), (CUBE_MM, CUBE_MM, CUBE_MM), None)
        stand = ((0.0, 0.0, 30.0), (CUBE_MM, CUBE_MM, 60.0), None)
        model = _model([block, part, stand], keep_out=[_held(0.0, 0.0, 60.0 + CUBE_MM)])

        under = _height_under(model, 0.0, 0.0)
        self.assertTrue(under is None or under < 10.0, model.render())

    def test_a_part_on_a_block_too_small_to_be_a_surface_stands_on_no_ground(self) -> None:
        """A 60 mm block's top round a 40 mm cube: its rim is no surface, and the lower part stays an obstacle."""
        block = ((0.0, 0.0, 30.0), (60.0, 60.0, 60.0), None)
        part = ((0.0, 0.0, 60.0 + CUBE_MM / 2.0), (CUBE_MM, CUBE_MM, CUBE_MM), None)
        model = _model([block, part], keep_out=[_held(0.0, 0.0, 60.0 + CUBE_MM)])

        under = _height_under(model, 0.0, 0.0)
        self.assertTrue(under is None or under < 10.0, model.render())

    def test_a_part_in_a_dense_pile_stands_on_the_bench_not_on_the_tops_round_it(self) -> None:
        """The owner's cell, 2026-10-15, 14:36 to 14:41: the tops of the cubes packed round a part as tall as they are
        were its ground, and the solid under it left its grasps the top few millimetres. A ground more than 10 mm over
        how low the part's own points reach stands beside it."""
        pile = [((x, y, CUBE_MM / 2.0), (CUBE_MM, CUBE_MM, CUBE_MM), None)
                for x in (-CUBE_MM, 0.0, CUBE_MM) for y in (-CUBE_MM, 0.0, CUBE_MM)]
        model = _model(pile, keep_out=[_held(0.0, 0.0, CUBE_MM)])

        under = _height_under(model, 0.0, 0.0)
        self.assertTrue(under is None or under < 10.0, model.render())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
