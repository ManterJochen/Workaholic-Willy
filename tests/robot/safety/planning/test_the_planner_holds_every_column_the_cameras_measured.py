"""cuRobo holds every column the cameras measured, not boxes merged to fit its slots (owner, 2026-10-15).

With ``safety.planning_world.perceived.voxel_field_mm`` the camera world sends cuRobo a distance field of the boxes of
every part's height map as the cameras measured them, before the slot budget merges them (``build_voxel_field``). The
boxes the field holds whole are not sent as boxes (``PerceivedWorld.field_held``): the field stands for them, so a bin
merged into one block to fit the slots keeps its inside free for the planner. The field is the camera's, named
``seen_field``: it goes aside with the camera's boxes where only they refuse a sample, and the exact guard decides
(``_curobo_perceived.SetAside``), and the guard holds every box the field was built of, so ``mesh_first`` may decide
with a field too.

Every scene is ray cast straight down through a pinhole (``tests/_seen_scenes.py``).
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.robot.safety.planning._curobo_perceived import SetAside
from src.robot.safety.planning._curobo_protocol import PERCEIVED_FIELD, PERCEIVED_PREFIX
from src.robot.safety.planning.live_world import CameraView, DepthSnapshot, LivePlannerWorld, refresh_planner_world
from src.robot.safety.planning.perceived import (
    DepthView,
    LinkCapsule,
    SelfEnvelope,
    VoxelField,
    WorldBuildTuning,
    build_perceived_boxes,
)
from src.robot.safety.planning.world import planner_cuboid
from tests._seen_scenes import K, LIMITS, covered_by, looking_down, open_bin, render, seen_points

#: A voxel of the field in these tests, millimetres.
_VOXEL = 10.0
_BIN = open_bin((0.0, -420.0), (300.0, 200.0), 40.0)
_CAMERA = looking_down(0.0, -420.0)


def _world(**tuning: Any) -> Any:
    depth = render(_CAMERA, _BIN)
    view = DepthView(surface_depth_mm=depth, intrinsics=K, camera_to_base=_CAMERA, name="overhead", timestamp=100.0)
    return build_perceived_boxes(views=(view,), limits=LIMITS,
                                 tuning=WorldBuildTuning(**{"voxel_field_mm": _VOXEL, **tuning}))


def _value(field: VoxelField, point: Any) -> float:
    nx, ny, nz = field.shape
    low = np.asarray(field.center_mm) - np.asarray(field.dims_mm) / 2.0
    index = np.floor((np.asarray(point, dtype=np.float64) - low) / field.voxel_size_mm).astype(int)
    return float(field.field.reshape(nx, ny, nz)[tuple(index)])


class TheFieldIsTheBoxesAsMeasuredTests(unittest.TestCase):
    def test_a_bin_merged_into_one_block_keeps_its_inside_free_in_the_field(self) -> None:
        roomy, tight = _world(), _world(max_boxes=len(_world().boxes) - 6)
        self.assertGreater(tight.merged_to_fit, 0, "the tight world merged nothing")
        np.testing.assert_array_equal(roomy.voxels.field, tight.voxels.field)
        inside = np.array([[0.0, -420.0, 10.0]])
        self.assertTrue(covered_by(tight.boxes, inside), "the merged block does not cover the inside")
        self.assertGreater(_value(tight.voxels, inside[0]), 0.0, "the field took the inside the cameras saw free")

    def test_every_point_the_cameras_saw_lies_in_an_obstacle_of_the_field(self) -> None:
        world = _world()
        field = world.voxels
        low = np.asarray(field.center_mm) - np.asarray(field.dims_mm) / 2.0
        seen = seen_points(_CAMERA, render(_CAMERA, _BIN))
        seen = seen[np.all((seen >= low) & (seen < low + np.asarray(field.dims_mm)), axis=1)]
        self.assertGreater(len(seen), 100)
        self.assertLessEqual(max(_value(field, point) for point in seen), 0.0)

    def test_the_boxes_the_field_holds_are_named(self) -> None:
        world = _world()
        self.assertTrue(world.field_held)
        self.assertTrue(set(world.field_held) <= {box.name for box in world.boxes})
        self.assertEqual(_world(voxel_field_mm=0.0).field_held, ())


class _Camera:
    def grab_surface_depth(self) -> DepthSnapshot:
        return DepthSnapshot(depth_mm=render(_CAMERA, _BIN), intrinsics=K, timestamp=100.0)


class _Scene:
    """A planner that takes the boxes, the meshes and the field in one request, as the sidecar does."""

    def __init__(self) -> None:
        import tempfile
        from pathlib import Path

        self.calls: list[dict[str, Any]] = []
        self.live_scene_path = str(Path(tempfile.gettempdir()) / f"willy_test_field_{id(self)}.npy")

    def set_scene(self, cuboids: Any, meshes: Any, voxels: Any) -> Any:
        from types import SimpleNamespace

        self.calls.append({"cuboids": list(cuboids), "voxels": voxels})
        count = None if voxels is None else int(np.load(voxels["path"]).shape[0])
        return SimpleNamespace(world_set=len(cuboids) + len(meshes or ()), voxels_set=count, reason="")


class ThePlannerIsSentTheFieldForTheBoxesItHoldsTests(unittest.TestCase):
    def test_the_boxes_the_field_holds_are_not_sent_and_the_field_is_named_the_cameras(self) -> None:
        source = LivePlannerWorld(
            cameras=(CameraView(name="overhead", depth_source=_Camera(), camera_to_base=_CAMERA),),
            declared=(planner_cuboid("support_plane", (0.0, 0.0, -25.0), (2000.0, 2000.0, 50.0)),),
            limits=LIMITS, tuning=WorldBuildTuning(voxel_field_mm=_VOXEL), max_age_ms=500.0,
        )
        body = SelfEnvelope(frames_mm=(np.eye(4),), capsules=(
            LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 100.0), radius_mm=50.0),))
        planner = _Scene()
        refresh = refresh_planner_world(source=source, client=planner, self_envelope=body,
                                        near_point_mm=(0.0, -420.0, 60.0), now=100.1)
        self.assertTrue(refresh.ok, refresh.reason)
        held = set(source.world_for(self_envelope=body, now=100.1).perceived.field_held)  # type: ignore[union-attr]
        sent = {box["name"] for box in planner.calls[-1]["cuboids"]}
        self.assertTrue(held)
        self.assertFalse(held & sent, "a box the field holds was sent as a box too")
        self.assertIn(PERCEIVED_FIELD, refresh.planner_seen)
        self.assertFalse(held & set(refresh.planner_seen))
        self.assertTrue(PERCEIVED_FIELD.startswith(PERCEIVED_PREFIX), "the field is the camera's")


class _Store:
    """The planner's storage as a judgement reads it: boxes and a field by name, each with its flag."""

    def __init__(self, names: "list[str]") -> None:
        self.flags = dict.fromkeys(names, True)

    def names(self) -> "list[str]":
        return list(self.flags)

    def enabled(self, name: str) -> bool:
        return self.flags[name]

    def set_enabled(self, name: str, enabled: bool) -> None:
        self.flags[name] = bool(enabled)


class TheFieldGoesAsideWithTheCamerasBoxesTests(unittest.TestCase):
    def test_the_field_and_the_boxes_go_aside_together_and_come_back(self) -> None:
        store = _Store(["support_plane", "seen_0", PERCEIVED_FIELD])
        with SetAside(store, ["seen_0", PERCEIVED_FIELD], prefix=PERCEIVED_PREFIX):
            self.assertEqual(store.flags, {"support_plane": True, "seen_0": False, PERCEIVED_FIELD: False})
        self.assertEqual(store.flags, {"support_plane": True, "seen_0": True, PERCEIVED_FIELD: True})


if __name__ == "__main__":
    unittest.main()
