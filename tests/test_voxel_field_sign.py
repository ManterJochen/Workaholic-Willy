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
pinned in ``tests/test_live_planner_world.py``.
"""

from __future__ import annotations

import unittest
from collections.abc import Sequence

import numpy as np

from src.robot.safety.planning.perceived import (
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


def _points(low: Sequence[float], high: Sequence[float], step_mm: float = 5.0) -> np.ndarray:
    axes = [np.arange(lo, hi + 1e-9, step_mm) for lo, hi in zip(low, high, strict=True)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([g.reshape(-1) for g in grid])


def _field(margin_mm: float) -> VoxelField:
    field = build_voxel_field(
        _points(_BLOCK_LOW, _BLOCK_HIGH),
        limits=_LIMITS,
        tuning=WorldBuildTuning(voxel_field_mm=_VOXEL_MM, margin_mm=margin_mm),
    )
    assert field is not None
    return field


def _value_at(field: VoxelField, point_mm: Sequence[float]) -> float:
    nx, ny, nz = field.shape
    low = np.array([_LIMITS.x_mm[0], _LIMITS.y_mm[0], float(_LIMITS.support_plane_top_mm or 0.0)])
    ix, iy, iz = (int(v) for v in np.floor((np.asarray(point_mm, dtype=np.float64) - low) / _VOXEL_MM))
    return float(field.field.reshape(nx, ny, nz)[ix, iy, iz])


class TheFieldSignTests(unittest.TestCase):
    def test_an_obstacle_reads_negative_and_free_space_positive(self) -> None:
        field = _field(margin_mm=0.0)

        self.assertLess(_value_at(field, _INSIDE), 0.0, "the planner reads negative as inside")
        self.assertGreater(_value_at(field, _FAR), 0.0, "positive is free space, or the grid is a wall")

    def test_the_margin_grows_the_obstacle(self) -> None:
        """A margin is clearance, so it has to make free space near the block read closer, not farther."""
        bare, grown = _field(margin_mm=0.0), _field(margin_mm=15.0)

        self.assertAlmostEqual(_value_at(grown, _NEAR), _value_at(bare, _NEAR) - 15.0, places=3)
        self.assertLess(_value_at(grown, _INSIDE), _value_at(bare, _INSIDE))


if __name__ == "__main__":
    unittest.main()
