"""The point-cloud corpus a grasp GENERATOR trains on — what goes in the cloud, and what does not.

Synthetic scenes, written as depth and mask PNGs and unprojected through the production camera code.
Nothing here needs Isaac or the real dataset; what it pins is the set of decisions that are easy to
get wrong once and never notice:

  * the ENVIRONMENT is in the cloud. The first version of the extractor unprojected only masked object
    pixels and discarded the background -- 63.6 % of pixels on a real scene, carrying the table, the
    bin walls and the arm. A generator that never sees a wall proposes grasps into one, and this
    project has measured 60.9 % of post-fusion finger collisions hitting exactly that.
  * it is CROPPED and COARSER. Taking the whole table at object resolution made 94.4 % of a scene's
    points table and pushed extraction from 0.62 to 8.63 s/scene.
  * `view_count` counts DISTINCT views, which is the only reason the fusion is not a `vstack`.
  * normals point at the cameras that SAW a point, not away from an object centroid -- the centroid
    rule is meaningless for a table.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from datagen.corpus.clouds import ARM_INSTANCE, ENVIRONMENT_INSTANCE, scene_cloud


class _Geometry:
    """The two attributes `scene_cloud` reads off a `SceneGeometry`."""

    def __init__(self, objects: dict[int, object]) -> None:
        self.objects = objects


def _write_arm(root: Path, name: str, size: int, rows: slice, cols: slice) -> None:
    """An ``<view>_arm.png`` marking a rectangle. 255 where the arm is, as the renderer writes it."""
    arm = np.zeros((size, size), dtype=np.uint8)
    arm[rows, cols] = 255
    Image.fromarray(arm).save(root / f"{name}_arm.png")


def _write_scene(root: Path, *, views: int = 2, size: int = 40,
                 object_depth_mm: int = 500, table_depth_mm: int = 600) -> dict:
    """A scene of ``views`` cameras looking straight down at one object on a table.

    The object is a square patch in the middle of the frame at a nearer depth; everything else is
    table. Every camera occupies the SAME pose: the first version of this helper offset them by a
    metre, which made them see different patches of table, so no voxel was ever seen twice and the
    observedness test failed against correct code. Co-located views are the honest unit test of "count
    DISTINCT views" -- whether two real cameras overlap is a fact about a rig, not about this function.
    """
    payload: dict = {"views": []}
    for index in range(views):
        depth = np.full((size, size), table_depth_mm, dtype=np.uint16)
        instances = np.zeros((size, size), dtype=np.uint8)
        depth[15:25, 15:25] = object_depth_mm
        instances[15:25, 15:25] = 1                    # instance id 0 -> mask value 1
        name = f"cam{index}"
        Image.fromarray(depth).save(root / f"{name}_depth.png")
        Image.fromarray(instances).save(root / f"{name}_instances.png")
        camera_to_base = np.eye(4)
        payload["views"].append({
            "name": name, "outcome": "rendered",
            "camera_to_base_mm": camera_to_base.tolist(),
            "intrinsics": [[400.0, 0.0, size / 2], [0.0, 400.0, size / 2], [0.0, 0.0, 1.0]],
        })
    (root / "scene.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


class EnvironmentTests(unittest.TestCase):
    def test_the_background_reaches_the_cloud_tagged_as_environment(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root)
            cloud = scene_cloud(root, payload, _Geometry({0: object()}),
                                environment_margin_mm=10_000.0)
        instance = cloud["instance_id"]
        self.assertIn(ENVIRONMENT_INSTANCE, instance.tolist())
        self.assertIn(0, instance.tolist())
        self.assertGreater(int((instance == ENVIRONMENT_INSTANCE).sum()), 0)

    def test_with_environment_false_reproduces_the_object_only_cloud(self) -> None:
        """The old behaviour is still reachable, so the two can be compared rather than argued about."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root)
            geometry = _Geometry({0: object()})
            without = scene_cloud(root, payload, geometry, with_environment=False)
            with_env = scene_cloud(root, payload, geometry, environment_margin_mm=10_000.0)
        self.assertNotIn(ENVIRONMENT_INSTANCE, without["instance_id"].tolist())
        np.testing.assert_allclose(
            without["points_mm"],
            with_env["points_mm"][with_env["instance_id"] != ENVIRONMENT_INSTANCE])

    def test_environment_further_than_the_margin_is_dropped(self) -> None:
        """A table point half a metre from any object cannot be touched by a grasp on it."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root)
            geometry = _Geometry({0: object()})
            wide = scene_cloud(root, payload, geometry, environment_margin_mm=10_000.0)
            tight = scene_cloud(root, payload, geometry, environment_margin_mm=1.0)
        wide_env = int((wide["instance_id"] == ENVIRONMENT_INSTANCE).sum())
        tight_env = int((tight["instance_id"] == ENVIRONMENT_INSTANCE).sum())
        self.assertGreater(wide_env, tight_env)
        # The OBJECT is never cropped -- only the environment is.
        self.assertEqual(int((wide["instance_id"] == 0).sum()),
                         int((tight["instance_id"] == 0).sum()))

    def test_the_environment_is_voxelised_more_coarsely_than_the_objects(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root)
            geometry = _Geometry({0: object()})
            fine = scene_cloud(root, payload, geometry, environment_voxel_mm=3.0,
                               environment_margin_mm=10_000.0)
            coarse = scene_cloud(root, payload, geometry, environment_voxel_mm=24.0,
                                 environment_margin_mm=10_000.0)
        self.assertGreater(int((fine["instance_id"] == ENVIRONMENT_INSTANCE).sum()),
                           int((coarse["instance_id"] == ENVIRONMENT_INSTANCE).sum()))
        self.assertEqual(int((fine["instance_id"] == 0).sum()),
                         int((coarse["instance_id"] == 0).sum()),
                         "the object grid must not move when the environment grid does")


class ArmChannelTests(unittest.TestCase):
    """The arm is its OWN class, not background.

    It is the one thing in a scene whose geometry changes completely between scenes, so a corpus that
    files it under "environment" hands a net a class with no consistent shape -- and the z-rotation
    augmentation has to leave it behind, because a rotated table is still a table while a rotated arm
    stands where no arm on a fixed base could.
    """

    def test_arm_pixels_become_their_own_instance_and_leave_the_environment(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root, views=1)
            geometry = _Geometry({0: object()})
            before = scene_cloud(root, payload, geometry, environment_margin_mm=10_000.0)
            _write_arm(root, "cam0", 40, slice(0, 8), slice(0, 8))
            after = scene_cloud(root, payload, geometry, environment_margin_mm=10_000.0)

        self.assertNotIn(ARM_INSTANCE, before["instance_id"].tolist())
        self.assertIn(ARM_INSTANCE, after["instance_id"].tolist())
        # The arm comes OUT of the environment, it is not added on top of it.
        self.assertLess(int((after["instance_id"] == ENVIRONMENT_INSTANCE).sum()),
                        int((before["instance_id"] == ENVIRONMENT_INSTANCE).sum()))
        self.assertEqual(int((after["instance_id"] >= 0).sum()),
                         int((before["instance_id"] >= 0).sum()),
                         "objects must not move when the arm is split out")

    def test_a_missing_arm_file_means_no_arm_pixels_not_an_error(self) -> None:
        """The renderer writes the file only when the arm is in frame -- the wrist camera never is."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root, views=1)
            cloud = scene_cloud(root, payload, _Geometry({0: object()}),
                                environment_margin_mm=10_000.0)
        self.assertNotIn(ARM_INSTANCE, cloud["instance_id"].tolist())

    def test_the_arm_does_not_shift_the_environment_crop(self) -> None:
        """The crop is measured from OBJECTS. An arm reaching across the table is not one."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root, views=1)
            geometry = _Geometry({0: object()})
            without = scene_cloud(root, payload, geometry, environment_margin_mm=50.0)
            _write_arm(root, "cam0", 40, slice(0, 8), slice(0, 8))
            with_arm = scene_cloud(root, payload, geometry, environment_margin_mm=50.0)
        self.assertEqual(int((without["instance_id"] == ENVIRONMENT_INSTANCE).sum()),
                         int((with_arm["instance_id"] == ENVIRONMENT_INSTANCE).sum())
                         + int((with_arm["instance_id"] == ARM_INSTANCE).sum())
                         - 0,
                         "splitting the arm out must not change WHICH points survive the crop")


class ObservednessTests(unittest.TestCase):
    def test_a_point_two_cameras_saw_carries_two(self) -> None:
        """The whole reason the fusion keeps view provenance instead of stacking."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root, views=2)
            two = scene_cloud(root, payload, _Geometry({0: object()}),
                              environment_margin_mm=10_000.0)
            payload_one = _write_scene(Path(name), views=1)
            one = scene_cloud(root, payload_one, _Geometry({0: object()}),
                              environment_margin_mm=10_000.0)
        # Co-located views, so every voxel is seen by both and the count is a count of VIEWS.
        self.assertEqual(int(two["view_count"].max()), 2)
        self.assertEqual(int(one["view_count"].max()), 1)

    def test_the_count_is_of_DISTINCT_views_not_of_points(self) -> None:
        """Two cameras contributing many points each must still count 2, never the point total."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root, views=2)
            cloud = scene_cloud(root, payload, _Geometry({0: object()}),
                                environment_margin_mm=10_000.0)
        self.assertLessEqual(int(cloud["view_count"].max()), 2)


class NormalTests(unittest.TestCase):
    def test_normals_are_unit_length(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root)
            cloud = scene_cloud(root, payload, _Geometry({0: object()}),
                                environment_margin_mm=10_000.0)
        valid = cloud["normal_valid"]
        if valid.any():
            np.testing.assert_allclose(
                np.linalg.norm(cloud["normals"][valid], axis=1), 1.0, atol=1e-5)

    def test_normals_face_the_cameras_that_saw_the_point(self) -> None:
        """The rule that replaced "outward from the object centroid", which a table does not have.

        Both cameras sit at z = 0 looking down +z, so every surface they see must have a normal with a
        NEGATIVE z: pointing back up the ray, out of the surface, towards the observer.
        """
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root)
            cloud = scene_cloud(root, payload, _Geometry({0: object()}),
                                environment_margin_mm=10_000.0)
        valid = cloud["normal_valid"]
        facing = cloud["normals"][valid][:, 2] < 0.0
        self.assertGreater(float(facing.mean()), 0.95,
                           "a normal pointing away from every camera that saw it is inside-out")


class DeterminismTests(unittest.TestCase):
    def test_two_extractions_of_one_scene_agree_exactly(self) -> None:
        """Measured byte-identical over 5 real scenes; pinned here so a dict order cannot creep in."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root)
            geometry = _Geometry({0: object()})
            first = scene_cloud(root, payload, geometry, environment_margin_mm=10_000.0)
            second = scene_cloud(root, payload, geometry, environment_margin_mm=10_000.0)
        for key in sorted(first):
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)


class EmptySceneTests(unittest.TestCase):
    def test_a_scene_with_no_rendered_view_returns_an_empty_cloud_rather_than_raising(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            payload = _write_scene(root)
            for view in payload["views"]:
                view["outcome"] = "failed"
            cloud = scene_cloud(root, payload, _Geometry({0: object()}))
        self.assertEqual(len(cloud["points_mm"]), 0)
        self.assertEqual(sorted(cloud),
                         ["instance_id", "normal_valid", "normals", "points_mm", "view_count"])


if __name__ == "__main__":
    unittest.main()
