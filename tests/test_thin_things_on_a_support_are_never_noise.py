"""A thin or small thing standing on a support or the bench is a thing, not noise (fix plan Track S, F1, test 6).

A cluster under ``min_points`` (12) thinned points is sensor noise. Before, a pin or a small square on the owner's mat hid
in the mat strip's cluster and was held; once the mat is a solid it stands alone, and the noise rule threw it away: a
3 mm pin 20 mm tall let the finger 1.0 mm into it (attack H1), a 12 mm cube 25 mm beside the mat was gone (H5). Now a
small cluster whose lowest point stands on a solid, a support's or the bench's, up to a cluster cell and a voxel (35 mm)
over its top, is a box whatever its count: a pin seen from straight above shows its top alone. What lies under the top
is the solid's, and speckle in mid-air past that is still noise.

The look is P1, recorded; the things drawn into it.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.robot.safety.planning.support_surfaces import find_supports
from src.robot.safety.planning.self_envelope import ROBOT_BASES
from tests import _cell_2026_10_01 as cell

#: The mat's outline in the owner's looks, BASE millimetres.
_MAT_X, _MAT_Y = (-411.3, 143.7), (-871.5, -509.1)


def _world(depth: np.ndarray) -> Any:
    owner = cell.owner_cell()
    snapshot = owner.world(cell.look("P1"), depth).world_for(
        self_envelope=owner.envelope(np.radians(cell.LOOK_0_DEG)), near_point_mm=(0.0, -650.0, 300.0))
    assert snapshot.perceived is not None, snapshot.reason
    return snapshot.perceived


def _held(world: Any, point: "tuple[float, float, float]", *, seen_only: bool = False) -> list[str]:
    out = []
    for box in world.boxes:
        if seen_only and box.kind != "seen":
            continue
        local = (np.asarray(point) - np.asarray(box.center_mm)) @ box.rotation_matrix
        if bool(np.all(np.abs(local) <= np.asarray(box.dims_mm) / 2.0)):
            out.append(box.name)
    return out


def _erasure_top(x: float, y: float) -> float:
    model = find_supports([cell.depth_view(cell.look("P1"))], limits=cell.OWNER_LIMITS,
                          tuning=cell.owner_tuning(support_surfaces=True, support_allowance_mm=2.0),
                          base=ROBOT_BASES["ur10"])
    tops = [float(s.erasure_top_at([[x, y]])[0]) for s in model.support_solids if s.covers_xy([[x, y]])[0]]
    assert tops, "no solid of the mat over the spot"
    return max(tops)


class ThinThingsOnTheMatAreHeldTests(unittest.TestCase):
    def test_a_3_mm_pin_20_mm_tall_on_the_mat_is_held(self) -> None:
        look = cell.look("P1")
        for x, y in ((40.0, -790.0), (60.0, -690.0), (-280.0, -770.0)):
            with self.subTest(at=(x, y)):
                mat = cell.local_reading(look, x, y, 0.0, 25.0)
                world = _world(cell.with_solids(look, cylinders=[(x, y, 1.5, mat, mat + 20.0)], seed=35))
                self.assertTrue(_held(world, (x, y, mat + 19.5), seen_only=True), "the pin left the world")

    def test_5_mm_squares_1_to_3_mm_over_the_solid_are_held(self) -> None:
        look = cell.look("P1")
        x, y = 60.0, -690.0
        mat = cell.local_reading(look, x, y, 0.0, 25.0)
        top = _erasure_top(x, y)
        for over in (1.0, 2.0, 3.0):
            with self.subTest(over_mm=over):
                height = top + over - mat
                world = _world(cell.with_solids(look, boxes=[((x, y, mat + height / 2.0), (5.0, 5.0, height), None)],
                                                seed=int(50 + over * 3)))
                self.assertTrue(_held(world, (x, y, top + over - 0.5)), "the square left the world")

    def test_a_12_mm_cube_25_mm_beside_the_mat_is_held(self) -> None:
        look = cell.look("P1")
        table = cell.level_table(look)
        mat = ((sum(_MAT_X) / 2.0, sum(_MAT_Y) / 2.0, 27.5), (_MAT_X[1] - _MAT_X[0], _MAT_Y[1] - _MAT_Y[0], 55.0), None)
        level_mat = cell.with_solids(look, boxes=[mat], depth_mm=table, noise_mm=1.3, seed=8)
        for gap in (25.0, 40.0, 60.0):
            with self.subTest(gap_mm=gap):
                x, y = -130.0, _MAT_Y[1] + gap + 6.0
                world = _world(cell.with_solids(look, boxes=[((x, y, 6.0), (12.0, 12.0, 12.0), None)],
                                                depth_mm=level_mat, seed=int(gap + 12)))
                self.assertTrue(_held(world, (x, y, 11.5), seen_only=True), "the cube beside the mat left the world")


class SpeckleIsStillNoiseTests(unittest.TestCase):
    def _speckled(self, height_over_mat: float) -> "tuple[Any, list[np.ndarray]]":
        """Two pixels the world reads, on the bare mat at (-340, -720), read ``height_over_mat`` over the mat's reading."""
        look = cell.look("P1")
        depth = np.array(look.surface_depth_mm, copy=True)
        origin, direction = cell.rays(look)
        rows, cols = depth.shape
        flat = np.arange(rows * cols).reshape(rows, cols)[::2, ::2].ravel()
        hit = origin[flat] + direction[flat] * depth.reshape(-1)[flat][:, None]
        index = flat[int(np.argmin(np.hypot(hit[:, 0] + 340.0, hit[:, 1] + 720.0)))]
        mat = cell.local_reading(look, -340.0, -720.0, 0.0, 25.0)
        points = []
        for pixel in (index, index + 4):
            t = (mat + height_over_mat - origin[pixel, 2]) / direction[pixel, 2]
            depth.reshape(-1)[pixel] = t
            points.append(origin[pixel] + direction[pixel] * t)
        return _world(depth), points

    def test_two_pixels_under_the_solids_top_are_no_box(self) -> None:
        world, points = self._speckled(3.0)
        for point in points:
            self.assertEqual(_held(world, tuple(point), seen_only=True), [])
            self.assertTrue(_held(world, tuple(point)), "what lies under the top is the solid's")

    def test_two_pixels_in_mid_air_are_noise(self) -> None:
        """Beyond a cluster cell and a voxel over the mat's solid: within that, a thin thing seen from above shows its
        top alone and is held, as it was when it joined the mat's own cluster."""
        world, points = self._speckled(80.0)
        for point in points:
            self.assertEqual(_held(world, tuple(point), seen_only=True), [])


if __name__ == "__main__":
    unittest.main()
