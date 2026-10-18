"""The distance field is built the way the planner reads it: negative inside an obstacle.

The code built it the other way round, positive inside, under a comment calling that measured, and no
test pinned either sign. Measured on the box with ``scripts/curobo/probe_live_world.py``
(``.commits/robot/51-the-world-and-the-hand.md``):

* a uniform field over the arm collides at -0.5 and -500 and is clear at +0.5 and +500, while the same
  fields far away and no field at all are clear;
* a one voxel plane written negative inside, in metres, touches the arm exactly where the same plane as
  a box does. The encoding the code wrote, positive inside and in millimetres, touched 395 mm early,
  which is when the grid's low edge reached the arm: every voxel that was not an obstacle read as one.

cuRobo's own voxel test builds its box field negative inside and positive outside. A field sent the old
way therefore made the whole reserved grid an obstacle, and the probe's chain passed for that reason:
the wall stopped the plan because the cell did.

`VoxelField` stays in millimetres like the rest of the module; the wire carries metres, and that half is
pinned in ``tests/robot/safety/planning/test_live_planner_world.py``. Since 2026-10-15 the field is built from the camera world's boxes as
measured (``tests/robot/safety/planning/test_the_planner_holds_every_column_the_cameras_measured.py``): each box carries the margin, the
field none of its own.
"""

from __future__ import annotations

import unittest
from collections.abc import Sequence

import numpy as np

from src.robot.safety.planning.perceived import (
    PerceivedBox,
    VoxelField,
    WorldBuildLimits,
    WorldBuildTuning,
    build_voxel_field,
)

#: A grid whose span is a whole number of voxels on every axis, so a point's voxel is plain arithmetic.
_LIMITS = WorldBuildLimits(
    x_mm=(0.0, 810.0), y_mm=(-405.0, 405.0), z_mm=(-50.0, 690.0), support_plane_top_mm=0.0
)
_VOXEL_MM = 30.0
_BLOCK_LOW = (300.0, -30.0, 0.0)
_BLOCK_HIGH = (360.0, 30.0, 90.0)
_INSIDE = (330.0, 0.0, 45.0)
_FAR = (60.0, -360.0, 600.0)
#: Four voxels beyond the block's high x face, in free space.
_NEAR = (360.0 + 4 * _VOXEL_MM + _VOXEL_MM / 2.0, 0.0, 45.0)


def _box(low: Sequence[float], high: Sequence[float], yaw_rad: float = 0.0) -> PerceivedBox:
    centre = tuple((a + b) / 2.0 for a, b in zip(low, high, strict=True))
    dims = tuple(b - a for a, b in zip(low, high, strict=True))
    return PerceivedBox(name="seen_0", center_mm=centre, dims_mm=dims, yaw_rad=yaw_rad, points=1,  # type: ignore[arg-type]
                        distance_mm=0.0)


def _field(*boxes: PerceivedBox) -> VoxelField:
    field = build_voxel_field(boxes or (_box(_BLOCK_LOW, _BLOCK_HIGH),), limits=_LIMITS,
                              tuning=WorldBuildTuning(voxel_field_mm=_VOXEL_MM))
    assert field is not None
    return field


def _value_at(field: VoxelField, point_mm: Sequence[float]) -> float:
    nx, ny, nz = field.shape
    low = np.array([_LIMITS.x_mm[0], _LIMITS.y_mm[0], float(_LIMITS.support_plane_top_mm or 0.0)])
    ix, iy, iz = (int(v) for v in np.floor((np.asarray(point_mm, dtype=np.float64) - low) / _VOXEL_MM))
    return float(field.field.reshape(nx, ny, nz)[ix, iy, iz])


class TheFieldSignTests(unittest.TestCase):
    def test_an_obstacle_reads_negative_and_free_space_positive(self) -> None:
        field = _field()

        self.assertLess(_value_at(field, _INSIDE), 0.0, "the planner reads negative as inside")
        self.assertGreater(_value_at(field, _FAR), 0.0, "positive is free space, or the grid is a wall")

    def test_a_voxel_that_touches_the_box_is_an_obstacle_and_one_clear_of_it_is_free(self) -> None:
        field = _field()
        touching = (360.0 + _VOXEL_MM / 2.0, 0.0, 45.0)
        self.assertLessEqual(_value_at(field, touching), 0.0, "a voxel the box's face reaches is the box's")
        # Four voxel centres from the voxel the face reaches into, which is the obstacle's nearest.
        self.assertAlmostEqual(_value_at(field, _NEAR), 4 * _VOXEL_MM, places=3)

    def test_every_point_of_a_turned_box_lies_in_the_field(self) -> None:
        box = _box((250.0, -60.0, 0.0), (410.0, 40.0, 120.0), yaw_rad=0.6)
        field = _field(box)
        c, s = np.cos(0.6), np.sin(0.6)
        for u in np.linspace(-80.0, 80.0, 9):
            for v in np.linspace(-50.0, 50.0, 7):
                point = (330.0 + c * u - s * v, -10.0 + s * u + c * v, 60.0)
                self.assertLessEqual(_value_at(field, point), 0.0, f"{point} of the turned box is free")

    def test_no_box_is_free_space_everywhere(self) -> None:
        field = build_voxel_field((), limits=_LIMITS, tuning=WorldBuildTuning(voxel_field_mm=_VOXEL_MM))
        assert field is not None
        self.assertEqual(field.occupied, 0)
        self.assertGreater(float(field.field.min()), 0.0)

    def test_the_field_is_the_transform_over_the_whole_grid(self) -> None:
        """The inside distances are taken over the occupied voxels' box and a free voxel round it (2026-10-15): the
        field is still the transform over the whole grid, boxes at the grid's edges and turned ones too."""
        from scipy import ndimage

        rng = np.random.default_rng(3)
        inner = [_box(_BLOCK_LOW, _BLOCK_HIGH)]
        for _ in range(5):
            low = rng.uniform((100.0, -300.0, 30.0), (600.0, 200.0, 400.0))
            inner.append(_box(low, low + rng.uniform(20.0, 150.0, 3), yaw_rad=float(rng.uniform(-1.0, 1.0))))
        edges = [_box((0.0, -405.0, 0.0), (90.0, -300.0, 690.0)), _box((700.0, 300.0, 600.0), (810.0, 405.0, 690.0))]
        for case, boxes in (("one box", inner[:1]), ("inside the grid", inner), ("at its edges", inner + edges)):
            with self.subTest(case=case):
                field = _field(*boxes)
                occupied = field.field.reshape(field.shape) < 0.0
                self.assertEqual(int(occupied.sum()), field.occupied)
                whole = (ndimage.distance_transform_edt(~occupied) - ndimage.distance_transform_edt(occupied)) * _VOXEL_MM
                np.testing.assert_allclose(field.field, whole.astype(np.float32).reshape(-1), rtol=0.0, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
