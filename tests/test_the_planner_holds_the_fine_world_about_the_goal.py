"""Near a motion's goal cuRobo holds the fine world the guard holds there (owner, 2026-10-15).

With a distance field (``perceived.voxel_field_mm``) and the fine world about a goal (``fine_world``), the field leaves
the fine square to a grid of its own (``VoxelField.near``, ``perceived.NEAR_VOXEL_MM``): every voxel of the field wholly
over the square is cleared, and the fine grid holds the guard's boxes over it, reaching ``perceived.NEAR_BAND_MM`` past
the square and over the highest of them, for cuRobo reads a grid at a sphere's centre and a centre off it as free.
Pinned on a part in a bin, ray cast straight down through a pinhole (``tests/_seen_scenes.py``):

* every point the cameras saw lies in an obstacle of one grid or the other;
* over the square the field is free and the fine grid holds the points; past it the field holds them and the fine grid
  is free; the fine grid stands as wide and as high as the band asks, never higher than the field;
* the room it gives: 10 mm over the part's top and beside its side the field of before held, and now neither grid does;
* the wire: the fine grid goes beside the field under ``near``, in a file of its own, named for the planner as the
  camera's; a channel of one grid a request refuses it rather than leave the square empty.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import ndimage

from src.robot.safety.planning._curobo_protocol import PERCEIVED_FIELD, PERCEIVED_NEAR_FIELD, PERCEIVED_PREFIX
from src.robot.safety.planning.live_world import (
    CameraView,
    DepthSnapshot,
    LivePlannerWorld,
    _write_voxel_field,
    refresh_planner_world,
)
from src.robot.safety.planning.perceived import (
    NEAR_BAND_MM,
    NEAR_VOXEL_MM,
    DepthView,
    LinkCapsule,
    SelfEnvelope,
    VoxelField,
    WorldBuildTuning,
    build_perceived_boxes,
)
from src.robot.safety.planning.world import planner_cuboid
from tests._seen_scenes import LIMITS, K, Solid, looking_down, open_bin, render, seen_points

_GOAL = (0.0, -420.0, 60.0)
_REACH = 150.0
_PART = Solid((0.0, -420.0, 25.0), (20.0, 20.0, 25.0))
_SCENE = [*open_bin((0.0, -420.0), (500.0, 200.0), 40.0), _PART]
_CAMERA = looking_down(0.0, -420.0)
_TUNING = WorldBuildTuning(voxel_field_mm=10.0, fine_reach_mm=_REACH)


def _world(tuning: WorldBuildTuning = _TUNING) -> Any:
    view = DepthView(surface_depth_mm=render(_CAMERA, _SCENE), intrinsics=K, camera_to_base=_CAMERA, name="overhead",
                     timestamp=100.0)
    return build_perceived_boxes(views=(view,), limits=LIMITS, tuning=tuning, near_point_mm=_GOAL)


def _read(field: VoxelField, points: np.ndarray) -> np.ndarray:
    """The field at ``points`` as cuRobo reads it, between voxel centres, ``inf`` off its grid."""
    nx, ny, nz = field.shape
    low = np.asarray(field.center_mm) - np.asarray(field.dims_mm) / 2.0
    at = (np.asarray(points, dtype=np.float64) - low) / field.voxel_size_mm - 0.5
    values = ndimage.map_coordinates(field.field.reshape(nx, ny, nz), at.T, order=1, mode="nearest")
    on = np.all((at >= -0.5) & (at <= np.array([nx, ny, nz]) - 0.5), axis=1)
    return np.where(on, values, np.inf)


def _beyond_square(points: np.ndarray) -> np.ndarray:
    """How far past the fine square each point stands in x or y, negative inside it."""
    return np.max(np.abs(points[:, :2] - np.asarray(_GOAL[:2])), axis=1) - _REACH


class TheFineGridAboutTheGoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = _world()
        cls.seen = seen_points(_CAMERA, render(_CAMERA, _SCENE))
        field = cls.world.voxels
        low = np.asarray(field.center_mm) - np.asarray(field.dims_mm) / 2.0
        cls.seen = cls.seen[np.all((cls.seen >= low) & (cls.seen < low + np.asarray(field.dims_mm)), axis=1)]

    def test_every_point_the_cameras_saw_lies_in_an_obstacle_of_one_grid_or_the_other(self) -> None:
        field = self.world.voxels
        self.assertIsNotNone(field.near)
        self.assertGreater(len(self.seen), 1000)
        both = np.minimum(_read(field, self.seen), _read(field.near, self.seen))
        self.assertLessEqual(float(both.max()), 0.0, self.seen[int(np.argmax(both))])

    def test_over_the_square_the_field_leaves_it_to_the_fine_grid(self) -> None:
        field = self.world.voxels
        over = self.seen[_beyond_square(self.seen) < -25.0]
        self.assertGreater(len(over), 1000)
        self.assertGreater(float(_read(field, over).min()), 0.0, "the field holds the square still")
        self.assertLessEqual(float(_read(field.near, over).max()), 0.0)

    def test_past_the_square_the_field_holds_what_it_held(self) -> None:
        field = self.world.voxels
        past = self.seen[_beyond_square(self.seen) > 10.0]
        self.assertGreater(len(past), 100)
        self.assertLessEqual(float(_read(field, past).max()), 0.0)
        self.assertGreater(float(_read(field.near, past).min()), 0.0, "the fine grid holds past the square")

    def test_the_fine_grid_reaches_the_band_past_the_square_and_over_the_highest_box(self) -> None:
        near = self.world.voxels.near
        self.assertEqual(near.voxel_size_mm, NEAR_VOXEL_MM)
        self.assertEqual(near.dims_mm[:2], (2.0 * (_REACH + NEAR_BAND_MM),) * 2)
        self.assertEqual(near.center_mm[:2], _GOAL[:2])
        solid = {box.name for box in self.world.supports.solids} if self.world.supports is not None else set()
        tops = [box.center_mm[2] + box.dims_mm[2] / 2.0 for box in self.world.guard_boxes if box.name not in solid
                and np.all(np.abs(np.asarray(box.center_mm[:2]) - _GOAL[:2]) <= np.asarray(box.dims_mm[:2]) / 2.0
                           + _REACH)]
        top = near.center_mm[2] + near.dims_mm[2] / 2.0
        self.assertGreaterEqual(top, max(tops) + NEAR_BAND_MM)
        self.assertLess(top, max(tops) + NEAR_BAND_MM + NEAR_VOXEL_MM)
        field = self.world.voxels
        self.assertLessEqual(top, field.center_mm[2] + field.dims_mm[2] / 2.0)

    def test_the_room_it_gives(self) -> None:
        """The field of before holds 10 mm over the part's top and 10 mm beside its side; neither grid holds there now."""
        before = _world(WorldBuildTuning(voxel_field_mm=10.0)).voxels
        self.assertIsNone(before.near)
        room = np.array([[0.0, -420.0, 60.0], [30.0, -420.0, 25.0]])
        self.assertLessEqual(float(_read(before, room).max()), 0.0, "the field of before left the room free already")
        field = self.world.voxels
        self.assertGreater(float(np.minimum(_read(field, room), _read(field.near, room)).min()), 0.0)

    def test_no_fine_world_no_fine_grid(self) -> None:
        self.assertIsNone(_world(WorldBuildTuning(fine_reach_mm=_REACH)).voxels)


class _Camera:
    def grab_surface_depth(self) -> DepthSnapshot:
        return DepthSnapshot(depth_mm=render(_CAMERA, _SCENE), intrinsics=K, timestamp=100.0)


def _source() -> LivePlannerWorld:
    return LivePlannerWorld(
        cameras=(CameraView(name="overhead", depth_source=_Camera(), camera_to_base=_CAMERA),),
        declared=(planner_cuboid("support_plane", (0.0, 0.0, -25.0), (2000.0, 2000.0, 50.0)),),
        limits=LIMITS, tuning=_TUNING, max_age_ms=500.0,
    )


_BODY = SelfEnvelope(frames_mm=(np.eye(4),), capsules=(
    LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 100.0), radius_mm=50.0),))


class _Scene:
    """A planner that takes the boxes, the meshes and the field in one request, as the sidecar does."""

    def __init__(self) -> None:
        import tempfile

        self.sent: dict[str, Any] = {}
        self.live_scene_path = os.path.join(tempfile.gettempdir(), f"willy_test_near_{id(self)}.npy")

    def set_scene(self, cuboids: Any, meshes: Any, voxels: Any) -> Any:
        self.sent = dict(voxels)
        count = sum(int(np.load(grid["path"]).shape[0]) for grid in (voxels, voxels.get("near")) if grid)
        return SimpleNamespace(world_set=len(cuboids) + len(meshes or ()), voxels_set=count, reason="")


class _OneGridAtATime:
    """A planner of the field channel alone, one grid a request."""

    def __init__(self) -> None:
        self.live_scene_path = _Scene().live_scene_path

    def set_world(self, cuboids: Any) -> int:
        return len(cuboids)

    def set_voxels(self, path: Any, **_: Any) -> int:
        return 1


class TheFineGridGoesBesideTheFieldTests(unittest.TestCase):
    def test_it_goes_under_near_in_a_file_of_its_own_and_is_named_the_cameras(self) -> None:
        planner = _Scene()
        refresh = refresh_planner_world(source=_source(), client=planner, self_envelope=_BODY, near_point_mm=_GOAL,
                                        now=100.1)
        self.assertTrue(refresh.ok, refresh.reason)
        near = planner.sent["near"]
        self.assertNotEqual(near["path"], planner.sent["path"])
        self.assertEqual(near["voxel_size_m"], NEAR_VOXEL_MM / 1000.0)
        cells = int(np.prod([round(d / 0.004) for d in near["dims_m"]]))
        self.assertEqual(int(np.load(near["path"]).shape[0]), -(-cells // 8), "eight voxels to a byte")
        self.assertIn(PERCEIVED_FIELD, refresh.planner_seen)
        self.assertIn(PERCEIVED_NEAR_FIELD, refresh.planner_seen)
        self.assertTrue(PERCEIVED_NEAR_FIELD.startswith(PERCEIVED_PREFIX), "the fine grid is the camera's")

    def test_the_writer_puts_its_occupancy_beside_the_fields_in_metres(self) -> None:
        field = _world().voxels
        payload = _write_voxel_field(_Scene(), field)
        for grid, sent in ((field, payload), (field.near, payload["near"])):
            values = np.load(sent["path"])
            self.assertEqual(values.dtype, np.uint8)
            np.testing.assert_array_equal(np.unpackbits(values, count=grid.occupancy.size).astype(bool), grid.occupancy)
            self.assertEqual(sent["pose"][:3], [c / 1000.0 for c in grid.center_mm])
            self.assertEqual(sent["dims_m"], [d / 1000.0 for d in grid.dims_mm])

    def test_a_channel_of_one_grid_a_request_refuses_it(self) -> None:
        refresh = refresh_planner_world(source=_source(), client=_OneGridAtATime(), self_envelope=_BODY,
                                        near_point_mm=_GOAL, now=100.1)
        self.assertFalse(refresh.ok)
        self.assertIn("one grid a request", refresh.reason)


if __name__ == "__main__":
    unittest.main()
